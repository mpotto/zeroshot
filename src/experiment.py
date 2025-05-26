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

from src.configs import configs
from src.defaults import defaults
from src.models import MiniCLIP, MiniVICReg
from src.data import get_multimodal_dataloaders

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
        if "imagenet_captions" in self.dataset:
            img_embed = self.cfg["data"]["img_embed"]
            txt_embed = self.cfg["data"]["txt_embed"]
            train_dataloader, test_dataloader, val_class_embeds = get_multimodal_dataloaders(
                batch_size, 
                rank, 
                img_embed,
                txt_embed,
                root=root,
            )
            self.val_class_embeds = None if val_class_embeds is None else torch.from_numpy(val_class_embeds)
            return train_dataloader, test_dataloader
        else:
            raise NotImplementedError(
                f"No dataset found in at path '{root}'!"
            )

    def get_model(self):
        model_cfg = self.cfg["model"]
        arch = model_cfg["architecture"]
        del model_cfg["architecture"]
        if arch == "clip":
            model = MiniCLIP(**model_cfg).float()
        elif arch == "vicreg":
            model = MiniVICReg(**model_cfg).float()
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
                self._compute_ssl_metrics(model, loader, out, split, eval_iters)
            else:
                raise NotImplementedError
        model.train()
        return out

    @torch.no_grad()
    def _compute_ssl_metrics(self, model, loader, out, split, eval_iters):
        for metric in ["loss"]:
            out[f"{split}_{metric}"] = 0.0
        denom = min(eval_iters, len(loader))
        it = 0
        for idx, X, Y in loader:
            if it >= eval_iters:
                break
            loss, logits = model(X.to(self.device), Y.to(self.device))
            out[f"{split}_loss"] += loss.item() / denom
            it += 1

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
