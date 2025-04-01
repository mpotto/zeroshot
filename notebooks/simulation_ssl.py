import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm

import sys
sys.path.extend([".", ".."])

from notebooks.simulation_gaussian import generate_distributions

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, n_layers, out_features):
        super(MLP, self).__init__()
        if n_layers > 0:
            self.proj = nn.Linear(in_features, hidden_size)
            self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers - 1)])
            self.out = nn.Linear(hidden_size, out_features)
        else:
            self.out = nn.Linear(in_features, out_features)
        self.n_layers = n_layers

    def forward(self, x):
        if self.n_layers > 0:
            x = F.relu(self.proj(x))
            for layer in self.layers:
                x = F.relu(layer(x))
        return self.out(x)

def clip_loss(logits):
    cx   = F.log_softmax(logits, dim=1)
    cy   = F.log_softmax(logits, dim=0)
    return -torch.mean(0.5 * torch.diagonal(cx) + 0.5 * torch.diagonal(cy))

# reach goal
def vicreg_loss(logits):
    pass

def get_loss(loss):
    if loss == "clip":
        return clip_loss
    else:
        raise NotImplementedError

class MiniCLIP(nn.Module):
    def __init__(
            self,
            in_features_img, 
            hidden_size_img, 
            n_layers_img,
            in_features_txt, 
            hidden_size_txt, 
            n_layers_txt,
            out_features, 
            loss="clip",
        ):
        super(MiniCLIP, self).__init__()
        self.image_encoder = MLP(in_features_img, hidden_size_img, n_layers_img, out_features)
        self.text_encoder  = MLP(in_features_txt, hidden_size_txt, n_layers_txt, out_features)
        # self.logit_scale = torch.nn.Parameter(0.1 * torch.randn(1)) # learnable parameter
        self.scale = 100.
        self.loss = get_loss(loss)

    def forward(self, x, z):

        # extract feature representations of each modality
        I_f = self.image_encoder(x)
        T_f = self.text_encoder(z)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(I_f)
        T_e = F.normalize(T_f)

        # scaled pairwise cosine similarities [n, n]
        logits = torch.matmul(I_e, T_e.T) * self.scale

        # symmetric loss function
        loss = self.loss(logits)

        return loss, logits

class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, x, z, y):
        self.x = x
        self.z = z
        self.y = y
        self.n = len(self.x)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i, self.x[i], self.z[i], self.y[i]
    

def run_ssl_experiment(p, a, b, a_, b_, setting, seed=123, device="cuda:0"):

    muX_0, muX_1 = setting["muX_0"], setting["muX_1"]
    d = setting["d"]

    # data distributions and parametrs
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


    torch.manual_seed(seed)

    # training data
    n_samples = 10000
    batch_size = 500
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    x = v[:, :len(muX_0)]
    z = v[:, len(muX_0):]
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).int()
    train_dataset = MultimodalEmbeddingDataset(x, z, y)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )

    # prompt data
    n_samples = 500
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    z0 = v[:n0, len(muX_0):]
    z1 = v[n0:, len(muX_0):]

    # evaluation data
    n_samples = 5000
    batch_size = 500
    n1 = int(n_samples * p)
    n0 = n_samples - n1
    v = torch.cat([mvn_0.rsample((n0,)), mvn_1.rsample((n1,))], dim=0)
    x = v[:, :len(muX_0)]
    z = v[:, len(muX_0):]
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).int()
    val_dataset = MultimodalEmbeddingDataset(x, z, y)
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
        "out_features": 2, 
    }
    model = MiniCLIP(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    n_epochs = 30
    for epoch in range(n_epochs):
        total_loss = 0
        for idx, x, z, y in train_dataloader:
            x = x.to(device)
            z = z.to(device)
            loss, logits = model(x, z)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"\t epoch {epoch + 1:02d}/{n_epochs:02d} loss: {total_loss / len(train_dataloader):0.5f}")

    # generate classifier
    with torch.no_grad():
        w0 = F.normalize(model.text_encoder(z0.to(device))).mean(dim=0)
        w1 = F.normalize(model.text_encoder(z1.to(device))).mean(dim=0)
        total_acc = 0
        for idx, x, z, y in val_dataloader:
            features = F.normalize(model.image_encoder(x.to(device)))
            scores = torch.stack([features @ w0, features @ w1])
            y_pred = torch.argmax(scores, dim=0).cpu()
            total_acc += ((y == y_pred).sum() / len(y)).item()
        avg_acc = total_acc / len(val_dataloader)

    return avg_acc