import math
import torch
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])

from src.simulation_ssl_multiclip import run_clip_experiment

p = 0.5
a = -0.1
b = 0.5
d = 2

# Markov Chain
setting_mc = {
    "muX_0": torch.ones(d)/2,
    "muX_1": -torch.ones(d)/2, 
    "d": d,
    "a": a, 
    "b": b,
    "rho": 1.0,
}

# No Markov Chain
setting_not_mc = {
    "muX_0": torch.ones(d)/2,
    "muX_1": -torch.ones(d)/2, 
    "d": d,
    "a": a, 
    "b": b,
    "rho": 0.0,
}

seeds = torch.arange(10)

for loss in ["clip", "doubly_centered"]:
    for prompt in ["pwy", "pzy"]:
        setting_mc["loss"] = loss
        setting_not_mc["loss"] = loss

        setting_mc["prompt"] = prompt
        setting_not_mc["prompt"] = prompt

        print(f"\t running loss {loss}, prompt {prompt}")

        for seed in seeds:
            print(f"\t running seed {seed}...")

            acc1s, acc2s = [], []

            acc1 = run_clip_experiment(p, setting_mc, seed=seed)
            acc1s.append(acc1)
            torch.save(torch.tensor(acc1s), f"notebooks/output/clip_accuracies_{d}_seed_{seed}_loss_{loss}_prompt_{prompt}_mc.pt")

            acc2 = run_clip_experiment(p, setting_not_mc, seed=seed)
            acc2s.append(acc2)
            torch.save(torch.tensor(acc2s), f"notebooks/output/clip_accuracies_{d}_seed_{seed}_loss_{loss}_prompt_{prompt}_nmc.pt")
    

