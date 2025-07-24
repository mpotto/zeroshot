import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Binomial
import math

from tqdm import tqdm

def generate_distributions(setting):
    # Embedding dimension
    d = setting["d"]
    a = setting["a"]
    b = setting["b"]
    rho = setting["rho"]

    # Mean
    muX_0, muX_1 = setting["muX_0"], setting["muX_1"]
    
    muZ_0, muZ_1 = rho * muX_0 / b, rho * muX_1 / b
    muW_0, muW_1 = rho * muX_0 / a, rho * muX_1 / a

    # Covariance
    CXX_0, CXX_1 = torch.eye(d), torch.eye(d)
    CZZ_0, CZZ_1 = torch.eye(d), torch.eye(d)
    CWW_0, CWW_1 = torch.eye(d), torch.eye(d)

    CXW_0, CXW_1 = a * torch.eye(d), a * torch.eye(d)
    CXZ_0, CXZ_1 = b * torch.eye(d), b * torch.eye(d)
    
    # rho = 1 satisfies the conditional independence
    CZW_0, CZW_1 = a/b * torch.eye(d), rho * a/b * torch.eye(d)  

    params = (muX_0, muX_1, muZ_0, muZ_1, muW_0, muW_1, 
              CXX_0, CXX_1, CZZ_0, CZZ_1, CWW_0, CWW_1, 
              CXW_0, CXW_1, CXZ_0, CXZ_1, CZW_0, CZW_1)
    return params

# Direct predictor computations
def compute_lpx(x, mvnX_0, mvnX_1, p):
    lp_1 = math.log(p)
    lp_0 = math.log(1 - p)
    numer = mvnX_1.log_prob(x) + lp_1
    denom = torch.logsumexp(torch.stack([mvnX_1.log_prob(x) + lp_1, mvnX_0.log_prob(x) + lp_0]), dim=0)
    return numer - denom

# Indirect predictor computations

# b(w) (notation in the write-up)
def compute_lpw(w, mvnW_0, mvnW_1, p):
    lp_1 = math.log(p)
    lp_0 = math.log(1 - p)
    numer = mvnW_1.log_prob(w) + lp_1
    denom = torch.logsumexp(torch.stack([mvnW_1.log_prob(w) + lp_1, mvnW_0.log_prob(w) + lp_0]), dim=0)
    return numer - denom

# a(z)
def compute_lpz(z, mvnZ_0, mvnZ_1, p):
    lp_1 = math.log(p)
    lp_0 = math.log(1 - p)
    numer = mvnZ_1.log_prob(z) + lp_1
    denom = torch.logsumexp(torch.stack([mvnZ_1.log_prob(z) + lp_1, mvnZ_0.log_prob(z) + lp_0]), dim=0)
    return numer - denom


def sample_W_given(z, p, setting, n_samples=1000, seed=123):
    muX_0, muX_1, muZ_0, muZ_1, muW_0, muW_1, CXX_0, CXX_1, CZZ_0, CZZ_1, CWW_0, CWW_1, CXW_0, CXW_1, CXZ_0, CXZ_1, CZW_0, CZW_1 = generate_distributions(setting)
    
    dist_0 = MultivariateNormal(loc=muW_0 + CZW_0 @ torch.linalg.solve(CZZ_0, z-muZ_0), covariance_matrix=CWW_0 - CZW_0 @ torch.linalg.solve(CZZ_0, CZW_0))
    dist_1 = MultivariateNormal(loc=muW_1 + CZW_1 @ torch.linalg.solve(CZZ_1, z-muZ_1), covariance_matrix=CWW_1 - CZW_1 @ torch.linalg.solve(CZZ_1, CZW_1))

    mvnZ_0 = MultivariateNormal(loc=muZ_0, covariance_matrix=CZZ_0)
    mvnZ_1 = MultivariateNormal(loc=muZ_1, covariance_matrix=CZZ_1)
    binom = Binomial(total_count=n_samples, probs=math.exp(compute_lpz(z, mvnZ_0, mvnZ_1, p)))

    torch.manual_seed(seed)
    n1 = binom.sample((1,)).int().item()
    n0 = n_samples - n1
    return torch.cat([dist_0.rsample((n0,)), dist_1.rsample((n1,))], dim=0)

# x -> E(a(Z)|X)(x)
def sample_Z_given(x, p, setting, n_samples=1000, seed=123):
    muX_0, muX_1, muZ_0, muZ_1, muW_0, muW_1, CXX_0, CXX_1, CZZ_0, CZZ_1, CWW_0, CWW_1, CXW_0, CXW_1, CXZ_0, CXZ_1, CZW_0, CZW_1 = generate_distributions(setting)

    dist_0 = MultivariateNormal(loc=muZ_0 + CXZ_0 @ torch.linalg.solve(CXX_0, x - muX_0), covariance_matrix=CZZ_0 - CXZ_0 @ torch.linalg.solve(CXX_0, CXZ_0))
    dist_1 = MultivariateNormal(loc=muZ_1 + CXZ_1 @ torch.linalg.solve(CXX_1, x - muX_1), covariance_matrix=CZZ_1 - CXZ_1 @ torch.linalg.solve(CXX_1, CXZ_1))

    mvnX_0 = MultivariateNormal(loc=muX_0, covariance_matrix=CXX_0)
    mvnX_1 = MultivariateNormal(loc=muX_1, covariance_matrix=CXX_1)
    binom = Binomial(total_count=n_samples, probs=math.exp(compute_lpx(x, mvnX_0, mvnX_1, p)))

    torch.manual_seed(seed)
    n1 = binom.sample((1,)).int().item()
    n0 = n_samples - n1
    return torch.cat([dist_0.rsample((n0,)), dist_1.rsample((n1,))], dim=0)

# Accuracy of Bayes predictor
def compute_bayes_accuracy(p, x, y, mvnX_0, mvnX_1):
    lpx = compute_lpx(x, mvnX_0, mvnX_1, p)
    y_pred = (lpx >= math.log(0.5)).int()
    return (y == y_pred).sum() / len(y)

# Three stage accuracy
def compute_three_stage_accuracy(p, x, y, setting, mvnW_0, mvnW_1, seed=123):
    px = []
    for x_ in tqdm(x):
        pz = []
        z = sample_Z_given(x_, p, setting, seed=seed, n_samples=10**2)
        for z_ in z:
            w = sample_W_given(z_, p, setting, seed=seed, n_samples=10**2)
            pz.append(torch.exp(compute_lpw(w, mvnW_0, mvnW_1, p)).mean())
        pz = torch.tensor(pz) 
        px.append(pz.mean())
    px = torch.tensor(px)
    y_pred = (px >= 0.5).int()
    return (y == y_pred).sum()/len(y)

def run_gaussian_experiment(p, setting, n_samples, seed=123):
    muX_0, muX_1, muZ_0, muZ_1, muW_0, muW_1, CXX_0, CXX_1, CZZ_0, CZZ_1, CWW_0, CWW_1, CXW_0, CXW_1, CXZ_0, CXZ_1, CZW_0, CZW_1 = generate_distributions(setting)
    
    mu0 = torch.cat([muX_0, muZ_0, muW_0])
    mu1 = torch.cat([muX_1, muZ_1, muW_1])
    cov0 = torch.cat(
        [
            torch.cat([CXX_0, CXZ_0, CXW_0], dim=1),
            torch.cat([CXZ_0, CZZ_0, CZW_0], dim=1),
            torch.cat([CXW_0, CZW_0, CWW_0], dim=1)
        ]
    )
    cov1 = torch.cat(
        [
            torch.cat([CXX_1, CXZ_1, CXW_1], dim=1),
            torch.cat([CXZ_1, CZZ_1, CZW_1], dim=1),
            torch.cat([CXW_1, CZW_1, CWW_1], dim=1)
        ]
    )
    mvn0 = MultivariateNormal(loc=mu0, covariance_matrix=cov0) 
    mvn1 = MultivariateNormal(loc=mu1, covariance_matrix=cov1)

    mvnX_0 = MultivariateNormal(loc=muX_0, covariance_matrix=CXX_0)
    mvnX_1 = MultivariateNormal(loc=muX_1, covariance_matrix=CXX_1)

    mvnZ_0 = MultivariateNormal(loc=muZ_0, covariance_matrix=CZZ_0)
    mvnZ_1 = MultivariateNormal(loc=muZ_1, covariance_matrix=CZZ_1)

    mvnW_0 = MultivariateNormal(loc=muW_0, covariance_matrix=CWW_0)
    mvnW_1 = MultivariateNormal(loc=muW_1, covariance_matrix=CWW_1)

    binom = Binomial(total_count=5000, probs=p)
    n1 = binom.sample((1,)).int().item()
    n0 = 5000 - n1
    x = torch.cat([mvnX_0.rsample((n0,)), mvnX_1.rsample((n1,))], dim=0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).int()
    acc1 = compute_bayes_accuracy(p, x, y, mvnX_0, mvnX_1)

    # compute accuracy (two stage/indirect)
    binom = Binomial(total_count=n_samples, probs=p)
    torch.manual_seed(seed)
    n1 = binom.sample((1,)).int().item()
    n0 = n_samples - n1
    x = torch.cat([mvnX_0.rsample((n0,)), mvnX_1.rsample((n1,))], dim=0)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).int()
    acc2 = compute_three_stage_accuracy(p, x, y, setting, mvnW_0, mvnW_1, seed=seed)

    return acc1.item(), acc2.item()
