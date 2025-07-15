import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Binomial
import math
from tqdm import tqdm

def generate_distributions(a, b, a_, b_, setting):

    muX_0, muX_1 = setting["muX_0"], setting["muX_1"]
    rho = setting["rho"]
    d = setting["d"]
    
    muZ_0 = (a_ / a) * muX_0 / rho
    muZ_1 = (b_ / b) * muX_1 / rho

    CZZ_0 = a * torch.eye(d)
    CZX_0 = rho * torch.eye(d) * a_
    CZZ_1 = b * torch.eye(d)
    CZX_1 = rho * torch.eye(d) * b_

    CXX_0 = (1 + rho ** 2 * a) * torch.eye(d)
    CXX_1 = (1 + rho ** 2 * b) * torch.eye(d)

    return  muZ_0, muZ_1, CZZ_0, CZX_0, CZZ_1, CZX_1, CXX_0, CXX_1

def sample_Z_given(x, muZ_0, CZX_0, CZZ_0, CXZ_0, muZ_1, CZX_1, CZZ_1, CXZ_1, CXX_0, CXX_1, p, setting, n_samples=1000, seed=123):
    muX_0, muX_1 = setting["muX_0"], setting["muX_1"]
    dist_0 = MultivariateNormal(loc=muZ_0 + CZX_0 @ torch.linalg.solve(CXX_0, x - muX_0), covariance_matrix=CZZ_0 - CZX_0 @ torch.linalg.solve(CXX_0, CXZ_0))
    dist_1 = MultivariateNormal(loc=muZ_1 + CZX_1 @ torch.linalg.solve(CXX_1, x - muX_1), covariance_matrix=CZZ_1 - CZX_1 @ torch.linalg.solve(CXX_1, CXZ_1))

    mvnX_0 = MultivariateNormal(loc=muX_0, covariance_matrix=CXX_0)
    mvnX_1 = MultivariateNormal(loc=muX_1, covariance_matrix=CXX_1)
    binom = Binomial(total_count=n_samples, probs=math.exp(compute_lpx(x, mvnX_0, mvnX_1, p)))

    torch.manual_seed(seed)
    n1 = binom.sample((1,)).int().item()
    n0 = n_samples - n1
    return torch.cat([dist_0.rsample((n0,)), dist_1.rsample((n1,))], dim=0)

def sample_X_given(z, muZ_0, muZ_1, muX_0, CXZ_0, CXX_0, CZX_0, muX_1, CXZ_1, CXX_1, CZX_1, CZZ_0, CZZ_1, p, n_samples=1000, seed=123):
    dist_0 = MultivariateNormal(loc=muX_0 + CXZ_0 @ torch.linalg.solve(CZZ_0, z - muZ_0), covariance_matrix=CXX_0 - CXZ_0 @ torch.linalg.solve(CZZ_0, CZX_0))
    dist_1 = MultivariateNormal(loc=muX_1 + CXZ_1 @ torch.linalg.solve(CZZ_1, z - muZ_1), covariance_matrix=CXX_1 - CXZ_1 @ torch.linalg.solve(CZZ_1, CZX_1))

    mvnZ_0 = MultivariateNormal(loc=muZ_0, covariance_matrix=CZZ_0)
    mvnZ_1 = MultivariateNormal(loc=muZ_1, covariance_matrix=CZZ_1)
    binom = Binomial(total_count=n_samples, probs=math.exp(compute_lpz(z, mvnZ_0, mvnZ_1, p)))

    torch.manual_seed(seed)
    n1 = binom.sample((1,)).int().item()
    n0 = n_samples - n1
    return torch.cat([dist_0.rsample((n0,)), dist_1.rsample((n1,))], dim=0)

def compute_lpxz(x, z, mvn_0, mvn_1, p):
    v = torch.cat([x, z], axis=1)
    lp_1 = math.log(p)
    lp_0 = math.log(1 - p)
    numer = mvn_1.log_prob(v) + lp_1
    denom = torch.logsumexp(torch.stack([mvn_1.log_prob(v) + lp_1, mvn_0.log_prob(v) + lp_0]), dim=0)

    return numer - denom

def compute_lpz(z, mvnZ_0, mvnZ_1, p):
    lp_1 = math.log(p)
    lp_0 = math.log(1 - p)
    numer = mvnZ_1.log_prob(z) + lp_1
    denom = torch.logsumexp(torch.stack([mvnZ_1.log_prob(z) + lp_1, mvnZ_0.log_prob(z) + lp_0]), dim=0)
    return numer - denom

def compute_lpx(x, mvnX_0, mvnX_1, p):
    lp_1 = math.log(p)
    lp_0 = math.log(1 - p)
    numer = mvnX_1.log_prob(x) + lp_1
    denom = torch.logsumexp(torch.stack([mvnX_1.log_prob(x) + lp_1, mvnX_0.log_prob(x) + lp_0]), dim=0)
    return numer - denom

def compute_cmi(mvn_0, mvn_1, mvnZ_0, mvnZ_1, n_samples, p, setting, seed=123):

    d = setting["d"]

    # compute test data
    torch.manual_seed(seed)
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    x = v[:, :d]
    z = v[:, d:]
    v_ = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    z_ =  v_[:, d:]

    lpxz = compute_lpxz(x, z, mvn_0, mvn_1, p)
    lpz = compute_lpz(z_, mvnZ_0, mvnZ_1, p)
    pxz = torch.exp(lpxz)
    pz = torch.exp(lpz)

    H_XZ = (-(pxz) * lpxz - (1 - pxz) * torch.log1p(-pxz)).mean()
    H_Z = (-(pz) * lpz - (1 - pz) * torch.log1p(-pz)).mean()

    return H_Z - H_XZ

def compute_information_density(x, y, z_, pz_, muX_0, muX_1, CXZ_0, CXZ_1, CZZ_0, CZZ_1, muZ_0, muZ_1, CXX_0, CXX_1, CZX_0, CZX_1):
    lp_1 = math.log(pz_)
    lp_0 = math.log(1 - pz_)

    mvnX_0_given_Z = MultivariateNormal(loc=muX_0 + CXZ_0 @ torch.linalg.solve(CZZ_0, z_ - muZ_0), covariance_matrix=CXX_0 - CXZ_0 @ torch.linalg.solve(CZZ_0, CZX_0))
    mvnX_1_given_Z = MultivariateNormal(loc=muX_1 + CXZ_1 @ torch.linalg.solve(CZZ_1, z_ - muZ_1), covariance_matrix=CXX_1 - CXZ_1 @ torch.linalg.solve(CZZ_1, CZX_1))
        
    numer = y * mvnX_1_given_Z.log_prob(x) + (1 - y) * mvnX_0_given_Z.log_prob(x)
    denom = torch.logsumexp(torch.stack([mvnX_1_given_Z.log_prob(x) + lp_1, mvnX_0_given_Z.log_prob(x) + lp_0]), dim=0)

    return torch.exp(numer - denom)

def compute_msc(muZ_0, muZ_1, muX_0, CXZ_0, CXX_0, CZX_0, muX_1, CXZ_1, CXX_1, CZX_1, CZZ_0, CZZ_1, n_samples_outer, n_samples_inner, p, seed=123):

    # generate z-values
    mvnZ_0 = MultivariateNormal(loc=muZ_0, covariance_matrix=CZZ_0)
    mvnZ_1 = MultivariateNormal(loc=muZ_1, covariance_matrix=CZZ_1)
    binom = Binomial(total_count=n_samples_outer, probs=p)
    torch.manual_seed(seed)
    n1 = binom.sample((1,)).int().item()
    n0 = n_samples_outer - n1
    z = torch.cat([mvnZ_0.rsample((n0,)), mvnZ_1.rsample((n1,))], dim=0)

    # generate data (n_samples_outer * n_samples_inner)
    Is = []
    pz = torch.exp(compute_lpz(z, mvnZ_0, mvnZ_1, p))
    for i in range(len(pz)):
        binom = Binomial(total_count=n_samples_inner, probs=pz[i])
        n1z = binom.sample((1,)).int().item()
        n0z = n_samples_inner - n1z
        yz = torch.cat([torch.zeros(n0z), torch.ones(n1z)]).int()
        xz = sample_X_given(z[i], muZ_0, muZ_1, muX_0, CXZ_0, CXX_0, CZX_0, muX_1, CXZ_1, CXX_1, CZX_1, CZZ_0, CZZ_1, p, n_samples=n_samples_inner, seed=seed)
        S = compute_information_density(xz, yz, z[i], pz[i], muX_0, muX_1, CXZ_0, CXZ_1, CZZ_0, CZZ_1, muZ_0, muZ_1, CXX_0, CXX_1, CZX_0, CZX_1)
        Is.append(((S - 1.) ** 2).mean())

    return torch.tensor(Is).mean()

def compute_bayes_accuracy(p, x, y, mvnX_0, mvnX_1):
    lpx = compute_lpx(x, mvnX_0, mvnX_1, p)
    y_pred = (lpx >= math.log(0.5)).int()
    return (y == y_pred).sum() / len(y)

def compute_two_stage_accuracy(p, x, y, muZ_0, CZX_0, CZZ_0, CXZ_0, muZ_1, CZX_1, CZZ_1, CXZ_1, CXX_0, CXX_1, mvnZ_0, mvnZ_1, setting, seed=123):
    px = []
    for x_ in x:
        z = sample_Z_given(x_, muZ_0, CZX_0, CZZ_0, CXZ_0, muZ_1, CZX_1, CZZ_1, CXZ_1, CXX_0, CXX_1, p, setting, n_samples=3000, seed=seed)
        px.append(torch.exp(compute_lpz(z, mvnZ_0, mvnZ_1, p)).mean())
    px = torch.tensor(px)
    y_pred = (px >= 0.5).int()
    return (y == y_pred).sum() / len(y)

def run_gaussian_experiment(p, a, b, a_, b_, setting, n_samples=10, seed=123, verbose=True):

    muX_0, muX_1 = setting["muX_0"], setting["muX_1"]

    # data distributions and parameters
    muZ_0,  muZ_1, CZZ_0, CZX_0, CZZ_1, CZX_1, CXX_0, CXX_1 = generate_distributions(a, b, a_, b_, setting)
    CXZ_0 = CZX_0.T
    CXZ_1 = CZX_1.T
    mu_0 = torch.cat([muX_0, muZ_0])
    mu_1 = torch.cat([muX_1, muZ_1])
    C_0 = torch.cat(
            [
                torch.cat([CXX_0, CXZ_0], dim=1),
                torch.cat([CZX_0, CZZ_0], dim=1)
            ], dim=0
        )
    C_1 = torch.cat(
        [
            torch.cat([CXX_1, CXZ_1], dim=1),
            torch.cat([CZX_1, CZZ_1], dim=1)
        ], dim=0
    )

    # samplers
    mvn_0 = MultivariateNormal(loc=mu_0, covariance_matrix=C_0)
    mvn_1 = MultivariateNormal(loc=mu_1, covariance_matrix=C_1)

    mvnX_0 = MultivariateNormal(loc=muX_0, covariance_matrix=CXX_0)
    mvnX_1 = MultivariateNormal(loc=muX_1, covariance_matrix=CXX_1)

    mvnZ_0 = MultivariateNormal(loc=muZ_0, covariance_matrix=CZZ_0)
    mvnZ_1 = MultivariateNormal(loc=muZ_1, covariance_matrix=CZZ_1)

    # test for conditional independence of X and Y given Z
    muX_given_Z_0 = muX_0 - CXZ_0 @ torch.linalg.solve(CZZ_0, muZ_0)
    muX_given_Z_1 = muX_1 - CXZ_1 @ torch.linalg.solve(CZZ_1, muZ_1)
    if verbose:
        print(f"\t conditional indep check first-order 1:  {torch.norm(muX_given_Z_0 - muX_given_Z_1).item():0.4f}")
        print(f"\t conditional indep check first-order 2:  {torch.norm(torch.linalg.solve(CZZ_0, CZX_0) - torch.linalg.solve(CZZ_1, CZX_1)).item():0.4f}")

    CXX_given_Z_0 = CXX_0 - CXZ_0 @ torch.linalg.solve(CZZ_0, CZX_0)
    CXX_given_Z_1 = CXX_1 - CXZ_1 @ torch.linalg.solve(CZZ_1, CZX_1)
    if verbose:
        print(f"\t conditional indep check second-order: {torch.norm(CXX_given_Z_0 - CXX_given_Z_1).item():0.4f}")
        print()

    I = compute_msc(muZ_0, muZ_1, muX_0, CXZ_0, CXX_0, CZX_0, muX_1, CXZ_1, CXX_1, CZX_1, CZZ_0, CZZ_1, 1000, 3000, p, seed=seed)

    # compute accuracy (bayes/direct)
    binom = Binomial(total_count=5000, probs=p)
    torch.manual_seed(seed)
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
    acc2 = compute_two_stage_accuracy(p, x, y, muZ_0, CZX_0, CZZ_0, CXZ_0, muZ_1, CZX_1, CZZ_1, CXZ_1, CXX_0, CXX_1, mvnZ_0, mvnZ_1, setting, seed=seed)
    
    return I.item(), acc1.item(), acc2.item()