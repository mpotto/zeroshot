import torch
import argparse

from src.experiment import ExperimentHelper

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
wd, mu, lr = helper.optim["weight_decay"], helper.optim["momentum"], helper.optim["lr"]

# Load data.
accumulation_steps_per_device = helper.accumulation_steps_per_device
batch_size = helper.effective_batch_size // (accumulation_steps_per_device * world_size)
train_loader, val_loader = helper.get_dataloaders(batch_size, rank)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, momentum=mu, weight_decay=wd)

# Run experiment.
model.train()
iter_num = 0
total_loss = 0.0
torch.manual_seed(0)
while iter_num < helper.max_iters * accumulation_steps_per_device:
    for idx, X, Z in train_loader:
        iter_num += 1
        helper.log_step(iter_num, model, [train_loader, val_loader])
        if iter_num >= helper.max_iters:
            break

        if is_ddp_run:
            model.require_backward_grad_sync = (
                iter_num % accumulation_steps_per_device == 0
            )

        # compute loss, potentially using variance reduction
        loss, logits = model(X.to(device), Z.to(device))
        total_loss += loss / accumulation_steps_per_device

        if iter_num % accumulation_steps_per_device == 0:
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss = 0.0
helper.end_experiment()
