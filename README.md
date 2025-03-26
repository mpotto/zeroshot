# zeroshot

This repository contains code and experiments for "From Pre-Training Foundation Models to Zero-Shot Prediction: Learning Paths, Prompt Complexity, and Residual Dependence". Please find instructions on software/hardware dependencies, reproducing all results from the manuscript below, and additional illustrations below.

## Abstract

A clever, modern approach to machine learning and AI takes a peculiar yet effective learning path involving two stages: from an upstream pre-training task using unlabeled multimodal data (foundation modeling), to a downstream task using prompting in natural language as a replacement for training data (zero-shot prediction). We cast this approach in a theoretical framework that allows us to identify the key quantities driving both its success and its pitfalls. We obtain risk bounds identifying the residual dependence lost when translating across modalities, the number and nature of prompts necessary for zero-shot prediction, and the discrepancy of this approach with classical single-stage machine learning.  

## Dependencies

We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM for ease of use. First, create a `conda` environment using the attached YAML file.
```
conda env create -f environment.yml
```
Note that PyTorch `2.1.0+cu121` was used for all experiments in this paper. Please use the PyTorch installation that is appropriate for your machine and CUDA distribution (see the [installation instructions](https://pytorch.org/get-started/locally/)).
For example, for CUDA 12.1, run:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Finally, for meta-prompting, we rely on the Hugging Face `transformers` module. To generate class-conditional prompts using LlaMA 3, first visit the [Hugging Face model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) to gain access.
In our script `scripts/generate_prompts.py`, it is assumed that there is a text file in the root of this repository called `token.txt`. This should contain a Hugging Face access token.

## Code

We outline the main features of this repository.

**Figures:**
The figures from the paper can be directly reproduced using notebooks of the form `notebooks/figure_*.ipynb`. See the list below.
| Figure      | Caption Header | Filename |
| ----------- | ----------- | ----------- |
| 3   | Results: Ideal Prompting | `notebooks/figure_ideal_prompts.ipynb`   |
| 4   | Results: Class-Conditional Prompting | `notebooks/figure_class_conditional_prompts.ipynb`   |

**Models and Data:** 
The three models and five standard benchmarks used are all managed via the `clip-benchmark` package. As such, all local files will be downloaded automatically when running an evaluation. See the [OpenCLIP results page](https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_classification_results.csv) for more information on these models.
| Model      | OpenCLIP Model Tag | Pre-Training Set Tag |
| ----------- | ----------- | ----------- |
| ResNet-50   | `RN50` | `yfcc15m`   |
| NLLB-CLIP   | `nllb-clip-base` | `v1`   |
| ViT-B/32    | `ViT-B-32` | `datacomp m s128m b4k`   |

The model tag indexes the subdirectories of `output/`. Similarly, the datasets are indexed by OpenCLIP data tags.

| Dataset      | Dataset Tag | 
| ----------- | ----------- | 
| [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)   | `dtd` | 
| [FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)   | `fgvc_aircraft` | 
| [Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)    | `flowers` | 
| [SUN397](https://vision.princeton.edu/projects/2010/SUN/)    | `sun397` | 
| [ImageNet-1k](https://image-net.org/download.php)    | `imagenet1k` | 

These tags are used for the subdirectories of `prompts/`, where all dataset-and-class specific prompts are stored in `.json` format. The dataset tags also index particular results within the model directories in `output`, for instance, `output/RN50/dtd`.
For ImageNet-Captions, we store a number of relevant files in `data`. The file `data/global_class_df.csv` lists a subset of 250 classes (listed in `data/imagenet_captions_train_c250.csv`) from ImageNet-1k in natural language, associated with a "global label" from 0 to 249. 
From these, subsets of 50 classes are selected and associated "local labels" from 0 to 49 at inference time. The `data/prompt_df.csv` file contains specific filenames (along with their associated classes and captions) from the ImageNet-1k dataset. 
These files are used to generate the ideal prompt embeddings; accordingly, the images are not used. On the other hand, `data/eval_df.csv` contains the data that is used to evaluation the zero-shot classification accuracy. Source code for managing data is contained in `src/data.py`.

**Prompts:** 
The prompts come in two forms.
- **Template-Based**: These are not downloaded locally, as they are loaded at evaluation time directly via CLIP Benchmark.
- **Class-Conditional:** In each directory of the form `prompts/{dataset}/`, the full list of LlaMA outputs are listed in `prompts/{dataset}/{dataset}_llama3_prompts_full.json`. For completeness, the GPT-3 outputs from [Pratt et. al. (ICCV, 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Pratt_What_Does_a_Platypus_Look_Like_Generating_Customized_Prompts_for_ICCV_2023_paper.pdf) are included as `prompts/{dataset}/{dataset}_prompts_full.json`, although only for comparison as they are not used in our experiments. For specific seeds used in experiments, see `prompts/{dataset}/sample_size_{n}_seed_{seed}.json`.

**Evaluations:** 
For the ImageNet-Captions experiments, they can be run using the format:
```
python scripts/evaluate_imagenet_captions_.py --model RN50 --device=0 --seed=0
```
This relies on caption embeddings that are already saved in `classifiers`, but can be recreated using `scripts/create_ideal_classifiers.py` and `scripts/create_template_classifiers.py`. Source code for managing ImageNet-Captions experiments is contained in `src/zeroshot.py`.
For the class-conditional prompting experiments, `scipts/run_zeroshot_complexity.py` generates the output for the baselines in Figure 4 whereas `scipts/run_zeroshot_complexity.py` generates the output for the curves. 
Note that you will have to specify a `ROOT` variable that contains the directory that you will use to store CLIP Benchmark output (primarily data). Then, an example run is:
```
CUDA_VISIBLE_DEVICES=0 python scripts/run_zeroshot_template.py --dataset=dtd --model=ViT-B-32 --batch_size=128
```


