import math
import torch
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])

from simulation_ssl import run_vicreg_experiment

p = 0.5
a = 5
b = 6
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

lam = 25

props = torch.load(f"notebooks/output/props_{d}.pt")

for seed in seeds:
    print(f"\t running seed {seed}...")
    acc3s = []
    for prop in tqdm(props):
        # print(f"control parameter = {prop:0.3f}")
        a_ = prop * a
        b_ = math.sqrt(prop) * b
        acc3 = run_vicreg_experiment(p, a, b, a_, b_, setting, lam, seed=seed)
        acc3s.append(acc3)
    torch.save(torch.tensor(acc3s), f"notebooks/output/vicreg_accuracies_{d}_seed_{seed}.pt")