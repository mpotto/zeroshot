import torch
import sys

sys.path.extend([".", ".."])

from src.simulation_gaussian_multiclip import run_gaussian_experiment

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

for seed in seeds:
    print(f"\t running seed {seed}...")
    
    acc1s, acc2s = [], []
    acc1, acc2 = run_gaussian_experiment(p, setting_mc, n_samples=10**2, seed=seed)
    acc1s.append(acc1)
    acc2s.append(acc2)

    torch.save(torch.tensor(acc1s), f"notebooks/output/bayes_accuracies_{d}_seed_{seed}_mc.pt")
    torch.save(torch.tensor(acc2s), f"notebooks/output/three_stage_accuracies_{d}_seed_{seed}_mc.pt")

    acc1s, acc2s = [], []
    acc1, acc2 = run_gaussian_experiment(p, setting_not_mc, n_samples=10**2, seed=seed)
    acc1s.append(acc1)
    acc2s.append(acc2)

    torch.save(torch.tensor(acc1s), f"notebooks/output/bayes_accuracies_{d}_seed_{seed}_nmc.pt")
    torch.save(torch.tensor(acc2s), f"notebooks/output/three_stage_accuracies_{d}_seed_{seed}_nmc.pt")
