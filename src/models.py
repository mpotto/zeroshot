import torch
import torch.nn as nn
import torch.nn.functional as F

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
        ):
        super(MiniCLIP, self).__init__()
        self.image_encoder = MLP(in_features_img, hidden_size_img, n_layers_img, out_features)
        self.text_encoder  = MLP(in_features_txt, hidden_size_txt, n_layers_txt, out_features)
        self.scale = 100.

    def _clip_loss(self, logits):
        cx   = F.log_softmax(logits, dim=1)
        cy   = F.log_softmax(logits, dim=0)
        return -torch.mean(0.5 * torch.diagonal(cx) + 0.5 * torch.diagonal(cy))

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
        loss = self._clip_loss(logits)

        return loss, logits
    
class MiniVICReg(nn.Module):
    def __init__(
            self,
            in_features_img, 
            hidden_size_img, 
            n_layers_img,
            in_features_txt, 
            hidden_size_txt, 
            n_layers_txt,
            out_features, 
            lam,
        ):
        super(MiniVICReg, self).__init__()
        self.image_encoder = MLP(in_features_img, hidden_size_img, n_layers_img, out_features)
        self.text_encoder  = MLP(in_features_txt, hidden_size_txt, n_layers_txt, out_features)

        self.gamma = 1.
        self.epsilon = 1e-4
        self.lam = lam
        self.mu = lam
        self.nu = 1.

        
    def forward(self, x, z):

        # extract feature representations of each modality
        I_e = self.image_encoder(x)
        T_e = self.text_encoder(z)

        # scaled pairwise cosine similarities [n, n]
        logits = (I_e, T_e)

        # eq (6) of https://arxiv.org/pdf/2105.04906
        loss = self.lam * self._s(I_e, T_e) + self.mu * (self._v(I_e) + self._v(T_e)) + self.nu * (self._c(I_e) + self._c(T_e))

        return loss, logits
    
    def _v(self, A):
        return F.relu(self.gamma - torch.sqrt(A.var(dim=0) + self.epsilon)).mean()
    
    def _s(self, A, B):
        return ((A - B) ** 2).sum() / len(A)
    
    def _c(self, A):
        cov = torch.cov(A.T, correction=0)
        return ((cov - torch.diag(cov)) ** 2).sum() / len(cov)