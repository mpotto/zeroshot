import math
import torch
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])

from notebooks.simulation_gaussian import run_gaussian_experiment

p = 0.5
a = 5
b = 6
props = torch.linspace(0.05, 1.0, 10)
seeds = torch.arange(10)

rho = 0.5
diff = 0.7
d = 2
muX_0 = torch.ones(d) * diff
muX_1 = -torch.ones(d) * diff

setting = {
    "rho": rho,
    "diff": diff,
    "d": d,
    "muX_0": torch.ones(d) * diff,
    "muX_1":  -torch.ones(d) * diff
}

for seed in seeds:
    print(f"\t running seed {seed}...")
    Is, acc1s, acc2s = [], [], []
    for prop in tqdm(props):
        # print(f"control parameter = {prop:0.3f}")
        a_ = prop * a
        b_ = math.sqrt(prop) * b
        I, acc1, acc2 = run_gaussian_experiment(p, a, b, a_, b_, setting, n_samples=2000, seed=seed, verbose=False)
        Is.append(I)
        acc1s.append(acc1)
        acc2s.append(acc2)
    torch.save(torch.tensor(Is), f"notebooks/output/msc_{d}_seed_{seed}.pt")
    torch.save(torch.tensor(acc1s), f"notebooks/output/bayes_accuracies_{d}_seed_{seed}.pt")
    torch.save(torch.tensor(acc2s), f"notebooks/output/two_stage_accuracies_{d}_seed_{seed}.pt")
torch.save(props, f"notebooks/output/props_{d}.pt")