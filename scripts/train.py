import torch
import argparse

from src.experiment import ExperimentHelper
from src.variance_reduction import compute_loss, compute_gradients

# TODO: gradient clipping 

# Option A: Use when running as a script.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="which dataset to run on, should be keys of dictionary in 'configs.py'",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="name of experiment for entry in 'configs.py'",
)
parser.add_argument(
    "--seed", type=int, default=0, help="seed for training run"
)
parser.add_argument("--device", type=str, default="cuda:0", help="gpu index")
args = parser.parse_args()
dataset, experiment_name, seed, device = args.dataset, args.experiment_name, args.seed, args.device

# Option B: Use with debugger.
# dataset, experiment_name, seed, device = "rebalanced_imagenet_captions_250k", "metaclip", 0, "cuda:0"

# Build model.
helper = ExperimentHelper(dataset, experiment_name, seed, device)
model = helper.get_model()

# Configure distributed data parallel.
is_ddp_run, device, rank, world_size = (
    helper.ddp,
    helper.device,
    helper.rank,
    helper.world_size,
)
wd, mu, vr = helper.optim["weight_decay"], helper.optim["momentum"], helper.variance_reduction

# Load data.
accumulation_steps_per_device = helper.accumulation_steps_per_device
batch_size = helper.effective_batch_size // (accumulation_steps_per_device * world_size)
train_loader, val_loader, quantization = helper.get_dataloaders(batch_size, rank)
if not (vr is None):
    vr['quantization'] = quantization

# Run experiment.
model.train()
if mu:
    momentum = [torch.zeros(param.shape).to(device) for param in model.parameters()]
else:
    momentum = [None for param in model.parameters()]
iter_num = 0
total_loss = 0.0
torch.manual_seed(0)
while iter_num < helper.max_iters * accumulation_steps_per_device:
    for idx, X, Y in train_loader:
        iter_num += 1
        helper.log_step(iter_num, model, [train_loader, val_loader])
        if iter_num >= helper.max_iters:
            break

        if is_ddp_run:
            model.require_backward_grad_sync = (
                iter_num % accumulation_steps_per_device == 0
            )

        # compute loss, potentially using variance reduction
        loss = compute_loss(model, idx, X.to(device), Y.to(device), vr=vr)
        total_loss += loss / accumulation_steps_per_device

        if iter_num % accumulation_steps_per_device == 0:
            # compute gradient, potentially using variance reduction
            parameters = list(model.parameters())
            gradients = compute_gradients(parameters, loss, vr=vr)
            if iter_num % accumulation_steps_per_device == 0:
                lr = helper.get_lr(iter_num)
                with torch.no_grad():
                    for param, g, mom in zip(parameters, gradients, momentum):
                        # weight decay update
                        if wd:
                            param *= 1 - wd * lr
                        # momentum update
                        if mu:
                            mom *= mu
                            mom += g
                            param -= lr * mom
                        else:
                            param -= lr * g

        # TODO: Check whether zero grad is necessary
        # model.zero_grad()
        total_loss = 0.0
helper.end_experiment()
