import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])

from src.simulation_gaussian_multiclip import generate_distributions
from src.models_multiclip import MiniCLIP

class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, x, z, w, y):
        self.x = x
        self.z = z
        self.y = y
        self.w = w
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i, self.x[i], self.z[i], self.w[i], self.y[i]
    

def run_clip_experiment(p, setting, seed=123, device="cuda:0"):

    muX_0, muX_1 = setting["muX_0"], setting["muX_1"]
    d = setting["d"]

    # data distributions and parametrs
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
    mvn_0 = MultivariateNormal(loc=mu0, covariance_matrix=cov0) 
    mvn_1 = MultivariateNormal(loc=mu1, covariance_matrix=cov1)

    torch.manual_seed(seed)

    # training data
    n_samples = 10000
    batch_size = 500
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    x = v[:, :len(muX_0)]
    z = v[:, len(muX_0):len(muX_0)+len(muZ_0)]
    w = v[:, len(muX_0)+len(muZ_0):]
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).int()
    train_dataset = MultimodalEmbeddingDataset(x, z, w, y)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )

    # prompt data
    n_samples = 500
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    z0 = v[:n0, len(muX_0):len(muX_0)+len(muZ_0)]
    w0 = v[:n0, len(muX_0)+len(muZ_0):]
    z1 = v[n0:, len(muX_0):len(muX_0)+len(muZ_0)]
    w1 = v[n0:, len(muX_0)+len(muZ_0):]

    # evaluation data
    n_samples = 5000
    batch_size = 500
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    x = v[:, :len(muX_0)]
    z = v[:, len(muX_0):len(muX_0)+len(muZ_0)]
    w = v[:, len(muX_0)+len(muZ_0):]
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).int()
    val_dataset = MultimodalEmbeddingDataset(x, z, w, y)
    val_dataloader = DataLoader(
        val_dataset, shuffle=True, batch_size=batch_size
    )

    model_cfg = {
        "in_features_img": d, 
        "hidden_size_img": 16, 
        "n_layers_img": 1,
        "in_features_txt": d, 
        "hidden_size_txt": 16, 
        "n_layers_txt": 1,
        "in_features_audio": d, 
        "hidden_size_audio": 16, 
        "n_layers_audio": 1,
        "out_features": 2, 
        "loss": setting["loss"]
    }
    model = MiniCLIP(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    n_epochs = 30
    for epoch in range(n_epochs):
        total_loss = 0
        for idx, x, z, w, y in train_dataloader:
            x = x.to(device)
            z = z.to(device)
            w = w.to(device)
            loss, logits = model(x, z, w)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"\t epoch {epoch + 1:02d}/{n_epochs:02d} loss: {total_loss / len(train_dataloader):0.5f}")

    # generate classifier
    with torch.no_grad():
        zz0 = F.normalize(model.text_encoder(z0.to(device))).mean(dim=0)
        zz1 = F.normalize(model.text_encoder(z1.to(device))).mean(dim=0)

        ww0 = F.normalize(model.audio_encoder(w0.to(device))).mean(dim=0)    
        ww1 = F.normalize(model.audio_encoder(w1.to(device))).mean(dim=0)

        total_acc = 0
        for idx, x, z, w, y in val_dataloader:
            if setting["prompt"] == "pzy":
                features = F.normalize(model.image_encoder(x.to(device)))
                scores = torch.stack([features @ zz0, features @ zz1])
            elif setting["prompt"] == "pwy":
                features = F.normalize(model.image_encoder(x.to(device)))
                scores = torch.stack([features @ ww0, features @ ww1])
            y_pred = torch.argmax(scores, dim=0).cpu()
            total_acc += ((y == y_pred).sum() / len(y)).item()
        avg_acc = total_acc / len(val_dataloader)

    return avg_acc
