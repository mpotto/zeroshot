import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, BertForMaskedLM

# TODO: Change this to where the models are on your machine!
MODEL_DIR = "/mnt/ssd/ronak/output/imagenet_captions_250k/"

####################################
# MODEL DEFINITIONS
####################################

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, n_layers, out_features):
        super(MLP, self).__init__()
        if n_layers > 0:
            self.proj = nn.Linear(in_features, hidden_size)
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)])
            self.out = nn.Linear(hidden_size, out_features)
        else:
            self.out = nn.Linear(in_features, out_features)
        self.n_layers = n_layers

    def forward(self, x):
        if self.n_layers > 0:
            x = F.relu(self.proj(x))
            for layer in self.layers:
                x = F.relu(layer(x))
        return self.out(x)


class MiniCLIP(nn.Module):
    def __init__(
            self,
            in_features_img, 
            hidden_size_img, 
            n_layers_img,
            in_features_txt, 
            hidden_size_txt, 
            n_layers_txt,
            out_features,
        ):
        super(MiniCLIP, self).__init__()
        self.image_encoder = MLP(in_features_img, hidden_size_img, n_layers_img, out_features)
        self.text_encoder  = MLP(in_features_txt, hidden_size_txt, n_layers_txt, out_features)
        self.scale = 100.

    def _clip_loss(self, logits):
        cx   = F.log_softmax(logits, dim=1)
        cy   = F.log_softmax(logits, dim=0)
        return -torch.mean(0.5 * torch.diagonal(cx) + 0.5 * torch.diagonal(cy))

    def forward(self, x, z):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(z)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.scale

        # symmetric loss function
        loss = self._clip_loss(logits)

        return loss, logits
    
class MiniVICReg(nn.Module):
    def __init__(
            self,
            in_features_img, 
            hidden_size_img, 
            n_layers_img,
            in_features_txt, 
            hidden_size_txt, 
            n_layers_txt,
            out_features, 
            lam,
        ):
        super(MiniVICReg, self).__init__()
        self.image_encoder = MLP(in_features_img, hidden_size_img, n_layers_img, out_features)
        self.text_encoder  = MLP(in_features_txt, hidden_size_txt, n_layers_txt, out_features)

        self.gamma = 1.
        self.epsilon = 1e-4
        self.lam = lam
        self.mu = lam
        self.nu = 1.

        
    def forward(self, x, z):

        # extract feature representations of each modality
        I_e = self.image_encoder(x)
        T_e = self.text_encoder(z)

        # scaled pairwise cosine similarities [n, n]
        logits = (I_e, T_e)

        # eq (6) of https://arxiv.org/pdf/2105.04906
        loss = self.lam * self._s(I_e, T_e) + self.mu * (self._v(I_e) + self._v(T_e)) + self.nu * (self._c(I_e) + self._c(T_e))

        return loss, logits
    
    def _v(self, A):
        return F.relu(self.gamma - torch.sqrt(A.var(dim=0) + self.epsilon)).mean()
    
    def _s(self, A, B):
        return ((A - B) ** 2).sum() / len(A)
    
    def _c(self, A):
        cov = torch.cov(A.T, correction=0)
        return ((cov - torch.diag(cov)) ** 2).sum() / len(cov)
    
####################################
# WRAPPERS FOR EVALUATION
####################################

class ModelWrapper(nn.Module):

    def __init__(self, foundation_img, foundation_txt, head):
        super(ModelWrapper, self).__init__()
        self.foundation_img = foundation_img
        self.foundation_txt = foundation_txt
        self.head = head
        self.scale = 100.

    def forward(self, image, text):
        I_f = self.encode_image(image)
        T_f = self.encode_text(text)
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)
        logits = torch.matmul(I_e, T_e.T) * self.scale

        return logits, logits.T
    
    def encode_image(self, image):
        image_features = self.foundation_img.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return self.head.image_encoder(image_features)

    def encode_text(self, text):
        text_features = self.foundation_txt.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return self.head.text_encoder(text_features)
    
def load_head_model(model_name):
    in_features_txt = 768 if "gpt" in model_name or "bert" in model_name else 512
    if "clip" in model_name:
        model_cfg = {
            "in_features_img": 512,
            "hidden_size_img": 256,
            "n_layers_img": 2,
            "in_features_txt": in_features_txt,
            "hidden_size_txt": 256,
            "n_layers_txt": 2,
            "out_features": 128,
        }
        model = MiniCLIP(**model_cfg)
    elif "clip" in model_name:
        model_cfg = {
            "in_features_img": 512,
            "hidden_size_img": 256,
            "n_layers_img": 2,
            "in_features_txt": in_features_txt,
            "hidden_size_txt": 256,
            "n_layers_txt": 2,
            "out_features": 128,
            "lam": 2, # does not matter for evaluation
        }
        model = MiniVICReg(**model_cfg)
    else:
        raise NotImplementedError

    output_dir = MODEL_DIR
    model.load_state_dict(torch.load(os.path.join(output_dir, f"{model_name}.pt")))
    return model

class SequenceModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokenized_texts):
        # dummy method, as only encode_text should be used
        return self.encode_text(tokenized_texts)

    def encode_text(self, tokenized_texts):
        device = tokenized_texts.get_device()
        input_ids, attn_mask = tokenized_texts[0], tokenized_texts[1]
        lens = torch.tensor([mask.sum().item() for mask in attn_mask])
        out = self.model(input_ids=input_ids.to(device), attention_mask=attn_mask.to(device))
        return torch.stack([out["hidden_states"][-1][i, 0:lens[i], :].mean(dim=0) for i in range(len(input_ids))])

class TokenizerWrapper:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, texts):
        # texts is a list of strings
        encoded_input = self.tokenizer(
            texts, 
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=32,
            # pad_to_max_length=True,
            truncation=True,
            padding='max_length',
            return_tensors="pt",  # Return pytorch tensors.return_tensors='pt'
            return_attention_mask=True,
        )
        # stack into two tensors for which the first are the ids and the second is the attention mask
        return torch.stack([encoded_input["input_ids"], encoded_input["attention_mask"]])

####################################
# LOADING FUNCTION
####################################

def load_miniclip(
        model_name: str = 'miniclip', 
        pretrained: str = 'laion2b_s34b_b79k', 
        cache_dir: str = None, 
        device="cpu",
    ):
    head_model = load_head_model(model_name)
    if "gpt" in model_name:
        seq_model = GPT2LMHeadModel.from_pretrained(f'gpt2', output_hidden_states=True)
        tokenizer = GPT2Tokenizer.from_pretrained(f'gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer = TokenizerWrapper(tokenizer)
        foundation_img, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir)
        foundation_txt = SequenceModelWrapper(seq_model)
        model = ModelWrapper(foundation_img, foundation_txt, head_model).to(device)
    elif "bert" in model_name:
        seq_model  = BertForMaskedLM.from_pretrained(
            "bert-base-uncased",
            output_attentions=False,
            output_hidden_states=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        tokenizer = TokenizerWrapper(tokenizer)
        foundation_img, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir)
        foundation_txt = SequenceModelWrapper(seq_model)
        model = ModelWrapper(foundation_img, foundation_txt, head_model).to(device)
    else:
        foundation_img, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', cache_dir=cache_dir)
        foundation_txt, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='datacomp_xl_s13b_b90k', cache_dir=cache_dir)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model = ModelWrapper(foundation_img, foundation_txt, head_model).to(device)
    return model, transform, tokenizer