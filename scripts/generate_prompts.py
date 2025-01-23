import transformers
import torch
import os
import json
import argparse
import time
import datetime

# setup
my_parser = argparse.ArgumentParser()
my_parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=['dtd', 'fgvc_aircraft', 'flowers', 'sun397', 'imagenet1k'],
)
my_parser.add_argument("--num_outputs", type=int, default=100)
my_args = my_parser.parse_args()

ROOT = "."
PROMPT_PATH = "prompts/"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.99

n_outputs = my_args.num_outputs
dataset = my_args.dataset

# build model
with open(os.path.join(ROOT, 'token.txt'), 'r') as f:
    HF_TOKEN = f.read()

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    token=HF_TOKEN
)


# build prompts
with open(os.path.join(PROMPT_PATH, f"{dataset}/{dataset}_prompts_full.json"), 'r') as f:
    templates = json.load(f)
with open(os.path.join(PROMPT_PATH, f"meta_prompts.json"), 'r') as f:
    meta_prompts = json.load(f)[dataset]

classnames = list(templates[dataset].keys())
print(f"dataset {dataset} has {len(classnames)} classes and {len(meta_prompts)} meta-prompts.")


instructions = """Please format your response as one that contains only lower case letters and no special characters (including new lines, bold, and any markdown artifacts) other than a period ('.') or commas (','). 
The response should be a single sentence ending in a period that is directed toward the final instruction in this message. Your sentence should be a minimum of three words and maximum of thirty."""

num_seeds = n_outputs // len(meta_prompts)
print(f"generating {num_seeds} prompts for each meta-prompt.")
ret = {class_.replace("(", "").replace(")", ""): [] for class_ in classnames}
for i, class_ in enumerate(classnames):
    prompts = [p.replace("{c}", class_) for p in meta_prompts]
    print(f"class {i + 1}/{len(classnames)}: {class_}")

    tic = time.time()

    for prompt in prompts:
        messages = [
            {"role": "user", "content": instructions + " " + prompt}
        ]
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            outputs = pipeline(
                messages,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=0.9,
                pad_token_id=128001
            )
            response = outputs[0]['generated_text'][1]['content']
            print(f"\t {response}")
            ret[class_.replace("(", "").replace(")", "")].append(response)

    toc = time.time()
    print(f"time taken: {datetime.datetime.fromtimestamp(toc - tic).strftime('%M:%S')}")
    print()

with open(os.path.join(PROMPT_PATH, f"{dataset}/{dataset}_llama3_prompts_full.json"), 'w') as f:
    templates = json.dump(ret, f, indent=4)