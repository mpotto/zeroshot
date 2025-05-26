import numpy as np
import torch
import random
import time
import datetime
import logging
import sys
import os
import pandas as pd
import json
import math
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from transformers import BertForSequenceClassification, AutoModelForCausalLM

from sklearn.metrics import classification_report, top_k_accuracy_score

from configs import configs
from defaults import defaults
from src.image_models import MyrtleNet, ResNet, ConvNet
from src.text_models import Transformer
from src.models import MiniCLIP, M
from src.image_data import get_image_dataloaders
from src.text_data import get_text_dataloaders, get_masked_dataloaders
from src.multimodal_data import get_multimodal_dataloaders
from src.variance_reduction import compute_loss, compute_gradients
from src.dro import configure_dro, compute_losses_no_grad

CACHE_DIR = "/mnt/ssd/ronak/models"

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ExperimentHelper:
    def __init__(self, dataset, experiment_name, seed, device):
        try:
            self.cfg = defaults[dataset].copy()
            changes = configs[dataset][experiment_name]
            for param_set in changes:
                if not (param_set in self.cfg):
                    self.cfg[param_set] = {}
                for key in changes[param_set]:
                    self.cfg[param_set][key] = changes[param_set][key]
        except KeyError:
            raise NotImplementedError(
                f"No configuration found for '{experiment_name}' in dataset '{dataset}' in configs.py!"
            )
        
        # TODO: This would be a section to change for other formats. We do not save this configuration
        self.variance_reduction = self.cfg['variance_reduction'] if 'variance_reduction' in self.cfg else None
        self.dro, self.compute_sample_weight = configure_dro(self.cfg['dro'], self.cfg['training']['batch_size']) if 'dro' in self.cfg else None, None

        # Expose what is necessary.
        self.dataset = dataset
        self.max_iters = self.cfg["training"]["max_iters"]
        self.optim = self.cfg["optim"]
        self.val_class_embeds = None


        (
            self.device,
            self.ddp,
            self.is_master_process,
            self.rank,
            self.world_size,
            self.accumulation_steps_per_device,
        ) = self._configure_ddp(device)
        self.rank = seed_offset = self.rank
        self.effective_batch_size = self.cfg["training"]["batch_size"]
        assert (
            self.effective_batch_size % self.accumulation_steps_per_device == 0
        ), "'grad_accumulation_steps' * 'world_size' must divide 'batch_size'"

        # Seed everything.
        random.seed(seed + seed_offset)
        np.random.seed(seed + seed_offset)
        torch.manual_seed(seed + seed_offset)
        torch.cuda.manual_seed_all(seed + seed_offset)

        # Create logger and logging/output directories.
        if self.is_master_process:
            save_dir = os.path.join(
                self.dataset,
                experiment_name,
                str(seed),
            )
            self.log_dir = os.path.join(self.cfg["training"]["log_dir"], save_dir)
            self.output_dir = os.path.join(self.cfg["training"]["output_dir"], save_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "config.json"), "w") as outfile:
                json.dump(self.cfg, outfile, indent=4)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(os.path.join(self.log_dir, "output.log")),
                    logging.StreamHandler(sys.stdout),
                ],
            )
            self.best_val_loss = torch.inf
            self.epoch_stats = []
            self.total_t0 = time.time()
            self.t0 = time.time()

    def _format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def _configure_ddp(self, device):
        ddp = int(os.environ.get("RANK", -1)) != -1
        grad_accumulation_steps = self.cfg["training"]["grad_accumulation_steps"]
        if ddp:
            init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device = f"cuda:{local_rank}"
            torch.cuda.set_device(device)
            is_master_process = local_rank == 0
        else:
            is_master_process = True
            local_rank = 0
            world_size = 1
        assert (
            grad_accumulation_steps % world_size == 0
        ), "'world_size' must divide 'grad_accumulation_steps'"
        accumulation_steps_per_device = grad_accumulation_steps // world_size

        return (
            device,
            ddp,
            is_master_process,
            local_rank,
            world_size,
            accumulation_steps_per_device,
        )

    def get_dataloaders(self, batch_size, rank):
        root = os.path.join(self.cfg["data"]["data_dir"], self.dataset)
        # unbalance = self.cfg["data"]["unbalance"]
        if not (self.variance_reduction is None):
            self.variance_reduction['quantization'] = {
                "x_labels":   np.load(os.path.join(root, f"quantization/{self.cfg['data']['x_labels']}")),
                "y_labels":   np.load(os.path.join(root, f"quantization/{self.cfg['data']['y_labels']}")),
                "x_marginal":   np.load(os.path.join(root, f"quantization/{self.cfg['data']['x_marginal']}")),
                "y_marginal":   np.load(os.path.join(root, f"quantization/{self.cfg['data']['y_marginal']}")),
            }

        if self.dataset in ["cifar10", "fashion_mnist", "stl10", "cifar100", "tiny_imagenet", "ub_fmnist"]:
            return get_image_dataloaders(
                batch_size, rank, root=root, quantization=self.variance_reduction['quantization']
            )
        elif self.dataset in ["sst2"]:
            return get_text_dataloaders(
                batch_size, rank, root=root, quantization=self.variance_reduction['quantization']
            )
        elif "imagenet_captions" in self.dataset:
            img_embed = self.cfg["data"]["img_embed"]
            txt_embed = self.cfg["data"]["txt_embed"]
            train_dataloader, test_dataloader, quantization, val_class_embeds = get_multimodal_dataloaders(
                batch_size, 
                rank, 
                img_embed,
                txt_embed,
                root=root, 
                # quantization=self.variance_reduction['quantization']
            )
            self.val_class_embeds = None if val_class_embeds is None else torch.from_numpy(val_class_embeds)
            return train_dataloader, test_dataloader, quantization
        elif self.dataset == "amazon":
            task = self.cfg["model"]["task"]
            objective = self.dro["objective"]
            train_dataloader, val_dataloader = get_masked_dataloaders(batch_size, task, self.dataset, objective=objective)
            return train_dataloader, val_dataloader
        else:
            raise NotImplementedError(
                f"No dataset found in at path '{root}'!"
            )

    def get_model(self):
        model_cfg = self.cfg["model"]
        arch = model_cfg["architecture"]
        del model_cfg["architecture"]
        if arch == "myrtle_net":
            model = MyrtleNet(**model_cfg).float()
        elif arch == "resnet":
            model = ResNet(**model_cfg).float()
        elif arch == "convnet":
            model = ConvNet(**model_cfg).float()
        elif arch == "transformer":
            model = Transformer(**model_cfg).float()
        elif arch == "miniclip":
            model = MiniCLIP(**model_cfg).float()
        elif arch == "bert_sequence_classification":
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=self.cfg['data']['num_labels'],
                output_attentions=False,
                output_hidden_states=False,
                cache_dir=CACHE_DIR,
            )
        elif arch == "baby_llama_causal_lm":
            model = AutoModelForCausalLM.from_pretrained(
                "timinar/baby-llama-58m",
                output_attentions=False,
                output_hidden_states=False,
                cache_dir=CACHE_DIR,
            )
        else:
            raise NotImplementedError(f"Unrecognized model architecture '{arch}'!")

        if isinstance(self.cfg["training"]["init_from"], int):
            # attempt to resume from a checkpoint.
            iter_num = self.cfg["init_from"]
            model.load_state_dict(
                torch.load(os.path.join(self.output_dir, f"ckpt_{iter_num}.pt"))
            )

        # Save a snapshot of the network architecture.
        if self.is_master_process:
            with open(os.path.join(self.log_dir, "model.txt"), "w") as f:
                print(model, file=f)
        model.to(self.device)

        if self.ddp:
            model = DDP(model, device_ids=[self.device])

        return model

    def get_lr(self, it):        
        optim = self.optim
        if optim['cosine_decay']:
            # 1) linear warmup for warmup_iters steps
            learning_rate = optim["lr"]
            # warmup_iters = int(0.01 * self.max_iters)
            warmup_iters = 0.0
            lr_decay_iters = self.max_iters
            min_lr = learning_rate / 10

            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        else:
            # decay every 50 epochs
            factor = 10 ** ((it * 256 / 14400) // 50)
            return optim["lr"] / factor

    # def get_optimizer(self, model):
    #     optim_cfg = self.cfg["optim_cfg"]
    #     algo = optim_cfg["algo"]
    #     optimizers = {"sgd": SGD, "adam": Adam}
    #     del optim_cfg["algo"]
    #     try:
    #         optimizer = optimizers[algo](model.parameters(), **optim_cfg)
    #         # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         #     optimizer,
    #         #     max_lr=optim_cfg["lr"] * 10,
    #         #     final_div_factor=10,
    #         #     steps_per_epoch=self.cfg["eval_interval"],
    #         #     total_steps=self.max_iters,
    #         #     pct_start=0.05,
    #         # )
    #     except KeyError:
    #         raise NotImplementedError(f"Unrecognized optimization algorithm '{algo}'!")
    #     # return optimizer, scheduler
    #     return optimizer

    def log_step(self, macro_iter_num, model, loaders):
        if (
            self.is_master_process
            and macro_iter_num % self.accumulation_steps_per_device == 0
        ):
            iter_num = macro_iter_num // self.accumulation_steps_per_device
            if iter_num % self.cfg["training"]["eval_interval"] == 0:
                if not iter_num == 0:
                    print()
                    logging.info(
                        f"Steps {iter_num - self.cfg['training']['eval_interval']:>5,} to {iter_num:>5,} took: {self._format_time(time.time() - self.t0)}."
                    )
                    print()

                    logging.info(
                        f"Evaluating using {self.cfg['training']['eval_iters']} batches..."
                    )
                    self.t0 = time.time()
                    # Compute evaluation metrics.
                    stats = self._compute_metrics(iter_num, model, loaders)
                    with open(
                        os.path.join(self.log_dir, f"step_{iter_num}.json"), "w"
                    ) as outfile:
                        json.dump(stats, outfile, indent=4)
                    self.epoch_stats.append(stats)
                    for metric in stats:
                        logging.info(f"    {metric}: {stats[metric]:0.4f}")
                    logging.info(
                        f"Evaluation took: {self._format_time(time.time() - self.t0)}."
                    )
                    # Checkpoint model.
                    if "imagenet_captions" in self.dataset or stats["validation_loss"] < self.best_val_loss:
                        logging.info(f"Saving checkpoint to '{self.output_dir}'...")
                        self.best_val_loss = stats["validation_loss"]
                        raw_model = model if not self.ddp else model.module
                        torch.save(
                            raw_model.state_dict(),
                            os.path.join(self.output_dir, f"ckpt_{iter_num}.pt"),
                        )

                if not iter_num == self.max_iters:
                    print()
                    logging.info(
                        f"======== Step {iter_num + 1:>5,} / {self.max_iters:>5,} ========"
                    )
                    logging.info("Training...")

                # Reset timer.
                self.t0 = time.time()

            elif iter_num % (self.cfg["training"]["eval_interval"] // 5) == 0 and not iter_num == 0:
                elapsed = format_time(time.time() - self.t0)
                logging.info(
                    f"    step {iter_num:>5,} / {self.max_iters:>5,}.    elapsed: {elapsed}."
                )

    @torch.no_grad()
    def _compute_metrics(self, iter_num, model, loaders):
        out = {"iter_num": iter_num}
        model.eval()
        eval_iters = self.cfg['training']["eval_iters"]
        out = {}
        for split, loader in zip(["train", "validation"], loaders):
            # TODO: Add language modeling.
            if "imagenet_captions" in self.dataset:
                self._compute_contrastive_metrics(model, loader, out, split, eval_iters)
            elif self.dataset == "amazon":
                task = self.cfg['model']['task']
                if task == "sequence_classification":
                    self._compute_sequence_classification_metrics(model, loader, out, split, eval_iters)
                elif task == "language_modeling":
                    self._compute_language_modeling_metrics(model, loader, out, split, eval_iters, num_labels=5)
                else:
                    raise NotImplementedError
            else:
                self._compute_classification_metrics(model, loader, out, split, eval_iters)
        if self.cfg['training']['track_variance']:
            for split, loader in zip(["train", "validation"], loaders):
                out[f"{split}_variance"] = self._compute_variance(model, loader, split)
        model.train()
        return out
    
    @torch.no_grad()
    def _compute_classification_metrics(self, model, loader, out, split, eval_iters):
        for metric in ["loss", "accuracy", "avg_precision", "min_precision", "avg_recall", "min_recall"]:
            out[f"{split}_{metric}"] = 0.0
        denom = min(eval_iters, len(loader))
        it = 0
        for idx, X, Y in loader:
            if it >= eval_iters:
                break
            Y = Y.to(self.device)
            loss, logits = model(X.to(self.device), Y)

            # accuracy
            Y_pred = torch.argmax(logits, dim=1)
            out[f"{split}_accuracy"] += (
                torch.sum((Y_pred == Y)) / len(Y)
            ).item() / denom

            # precision and recall
            report = classification_report(Y.cpu(), Y_pred.cpu(), zero_division=0.0, output_dict=True)
            labels = []
            for key in list(report.keys()):
                if key not in ['accuracy', 'marco avg', 'weighted avg']:
                    labels.append(key)
            precision = np.array([report[label]['precision'] for label in labels])
            recall = np.array([report[label]['recall'] for label in labels])
            out[f"{split}_avg_precision"] += precision.mean() / denom
            out[f"{split}_min_precision"] += precision.min() / denom
            out[f"{split}_avg_recall"] += recall.mean() / denom
            out[f"{split}_min_recall"] += recall.min() / denom

            out[f"{split}_loss"] += loss.item() / denom
            it += 1

    @torch.no_grad()
    def _compute_sequence_classification_metrics(self, model, loader, out, split, eval_iters):
        for metric in ["loss", "accuracy", "avg_precision", "min_precision", "avg_recall", "min_recall"]:
            out[f"{split}_{metric}"] = 0.0
        denom = min(eval_iters, len(loader))
        it = 0
        for batch in loader:
            if it >= eval_iters:
                break
            idx = batch[0].to(self.device)
            input_ids = batch[1].to(self.device)
            input_mask = batch[2].to(self.device)
            labels = batch[3].to(self.device)

            output = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                labels=labels,
                return_dict=True,
            )
            loss, logits = output.loss, output.logits

            # accuracy
            Y_pred = torch.argmax(logits, dim=1)
            Y = labels
            out[f"{split}_accuracy"] += (
                torch.sum((Y_pred == Y)) / len(Y)
            ).item() / denom

            # precision and recall
            report = classification_report(Y.cpu(), Y_pred.cpu(), zero_division=0.0, output_dict=True)
            labels = []
            for key in list(report.keys()):
                if key not in ['accuracy', 'marco avg', 'weighted avg']:
                    labels.append(key)
            precision = np.array([report[label]['precision'] for label in labels])
            recall = np.array([report[label]['recall'] for label in labels])
            out[f"{split}_avg_precision"] += precision.mean() / denom
            out[f"{split}_min_precision"] += precision.min() / denom
            out[f"{split}_avg_recall"] += recall.mean() / denom
            out[f"{split}_min_recall"] += recall.min() / denom

            out[f"{split}_loss"] += loss.item() / denom
            it += 1

    @torch.no_grad()
    def _compute_language_modeling_metrics(self, model, loader, out, split, eval_iters, num_labels=5):
        for metric in ["loss"]:
            out[f"{split}_{metric}"] = 0.0
        for label_ in range(num_labels):
            out[f"{split}_loss_label_{label_}"] = 0.0

        denom = min(eval_iters, len(loader))
        it = 0
        label_batches = {label_: 0 for label_ in range(num_labels)}
        for batch in loader:
            if it >= eval_iters:
                break
            idx = batch[0].to(self.device)
            input_ids = batch[1].to(self.device)
            input_mask = batch[2].to(self.device)
            labels = batch[3].to(self.device)

            output = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                labels=input_ids,
                return_dict=True,
            )
            loss, logits = output.loss, output.logits
            out[f"{split}_loss"] += loss.item() / denom

            losses = compute_losses_no_grad(logits, input_ids, input_mask)

            for label_ in range(num_labels):
                label_idx = torch.arange(len(input_ids)).to(self.device)[labels==label_]
                if len(label_idx) > 0:
                    label_batches[label_] += 1
                    # output = model(
                    #     input_ids=input_ids[label_idx],
                    #     attention_mask=input_mask[label_idx],
                    #     labels=input_ids[label_idx],
                    #     return_dict=True,
                    # )
                    # loss, logits = output.loss, output.logits
                    out[f"{split}_loss_label_{label_}"] += torch.mean(losses[label_idx]).item()

            it += 1

        for label_ in range(num_labels):
            out[f"{split}_loss_label_{label_}"] /= label_batches[label_]

    @torch.no_grad()
    def _compute_contrastive_metrics(self, model, loader, out, split, eval_iters):
        for metric in ["loss", "zero_shot_top_1", "zero_shot_top_2", "zero_shot_top_5"]:
            out[f"{split}_{metric}"] = 0.0
        denom = min(eval_iters, len(loader))
        it = 0
        if split == "validation" and not (self.val_class_embeds is None):
            class_embeds = self.val_class_embeds.to(self.device)
            class_encodings_T = torch.nn.functional.normalize(model.text_encoder(class_embeds)).T
            for idx, X, Y, Z in loader:
                if it >= eval_iters:
                    break

                # compute standard loss
                loss, logits = model(X.to(self.device), Y.to(self.device))
                out[f"{split}_loss"] += loss.item() / denom

                # compute zero-shot accuracy
                img_encodings = torch.nn.functional.normalize(model.image_encoder(X.to(self.device)))
                class_scores = torch.matmul(img_encodings, class_encodings_T).cpu()
                out[f"{split}_zero_shot_top_1"] += top_k_accuracy_score(Z, class_scores, k=1) / denom
                out[f"{split}_zero_shot_top_2"] += top_k_accuracy_score(Z, class_scores, k=2) / denom
                out[f"{split}_zero_shot_top_5"] += top_k_accuracy_score(Z, class_scores, k=5) / denom

                it += 1
        else:
            for idx, X, Y in loader:
                if it >= eval_iters:
                    break
                loss, logits = model(X.to(self.device), Y.to(self.device))
                out[f"{split}_loss"] += loss.item() / denom
                it += 1
    
    @torch.no_grad()
    def _compute_variance(self, model, loader, split, max_iters=200):
        device = self.device
        vr = self.variance_reduction

        # estimate full batch gradient
        means = [torch.zeros(param.shape).to(device) for param in model.parameters()]
        it = 0
        while it < max_iters:
            if split == "validation" and not (self.val_class_embeds is None):
                for idx, X, Y, Z in loader:
                    if it >= max_iters:
                        break
                    Y = Y.to(self.device)
                    with torch.enable_grad():
                        model.zero_grad()
                        # no variance reduction, as we will take enough batches to compute the full quantity
                        loss = compute_loss(model, idx, X.to(device), Y.to(device), vr={})
                        gradients = compute_gradients(list(model.parameters()), loss, vr={})
                    for mean, grad in zip(means, gradients):
                        mean += grad / max_iters
                    model.zero_grad()
                    it += 1
            else:
                for idx, X, Y in loader:
                    if it >= max_iters:
                        break
                    Y = Y.to(self.device)
                    with torch.enable_grad():
                        model.zero_grad()
                        # no variance reduction, as we will take enough batches to compute the full quantity
                        loss = compute_loss(model, idx, X.to(device), Y.to(device), vr={})
                        gradients = compute_gradients(list(model.parameters()), loss, vr={})
                    for mean, grad in zip(means, gradients):
                        mean += grad / max_iters
                    model.zero_grad()
                    it += 1
        
        # estimate variance of stochastic gradients
        variance = 0
        it = 0
        while it < max_iters:
            if split == "validation" and not (self.val_class_embeds is None):
                for idx, X, Y, Z in loader:
                    if it >= max_iters:
                        break
                    Y = Y.to(self.device)
                    with torch.enable_grad():
                        model.zero_grad()
                        loss = compute_loss(model, idx, X.to(device), Y.to(device), vr=vr)
                        gradients = compute_gradients(list(model.parameters()), loss, vr=vr)
                    for mean, grad in zip(means, gradients):
                        variance += torch.norm(grad - mean) ** 2 / max_iters
                    it += 1
            else:
                for idx, X, Y in loader:
                    if it >= max_iters:
                        break
                    Y = Y.to(self.device)
                    with torch.enable_grad():
                        model.zero_grad()
                        loss = compute_loss(model, idx, X.to(device), Y.to(device), vr=vr)
                        gradients = compute_gradients(list(model.parameters()), loss, vr=vr)
                    for mean, grad in zip(means, gradients):
                        variance += torch.norm(grad - mean) ** 2 / max_iters
                    it += 1

        return variance.item()
    
    def resample(self, idx, x, y):
        nlabels = self.cfg['model']['n_classes']
        batch_size = self.cfg['training']['batch_size']

        size = int(batch_size / nlabels)
        remain = batch_size - size * nlabels
        ind = []
        for label in range(nlabels):
            _size = size
            if remain > 0:
                _size += 1
                remain -= 1
            candidate = (y == label).nonzero(as_tuple=True)[0]
            if len(candidate) > 0:
                ind += list(candidate[torch.multinomial(torch.ones(len(candidate)), _size, replacement=True)])
        return idx[ind], x[ind], y[ind]

    def end_experiment(self):
        if self.is_master_process:
            print()
            logging.info(
                f"Training complete! Total time: {format_time(time.time() - self.total_t0)}"
            )

            # Save epoch metrics in readable format.
            df = pd.DataFrame(self.epoch_stats)
            with open(os.path.join(self.log_dir, "epoch_stats.csv"), "w") as f:
                df.to_csv(f, index=False)
        if self.ddp:
            destroy_process_group()
