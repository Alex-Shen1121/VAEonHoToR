import math

import torch
import torch.nn as nn


class HoToR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.U = nn.Parameter(torch.empty(user_size, dim))
        self.V = nn.Parameter(torch.empty(item_size, dim))
        self.biasV = nn.Parameter(torch.empty(item_size))
        nn.init.xavier_normal_(self.U.data)
        nn.init.xavier_normal_(self.V.data)
        nn.init.normal_(self.biasV, 1e-3)
        self.weight_decay = weight_decay

    def forward(self, u, i, r_ui, j):
        biasI = self.biasV[i]
        biasJ = self.biasV[j]
        u = self.U[u, :]
        i = self.V[i, :]
        j = self.V[j, :]
        x_ui = torch.mul(u, i).sum(dim=1) + biasI
        x_uj = torch.mul(u, j).sum(dim=1) + biasJ
        r_uij = x_ui - x_uj
        barr_ui = torch.tensor(
            [1 if r == 5 else (math.pow(2, r) - 1) / math.pow(2, 5) for r in r_ui],
            requires_grad=False
        )
        r_uij = r_uij * barr_ui
        log_prob = r_uij.sigmoid().log().sum()
        regularization = self.weight_decay * (
                u.norm(dim=1).pow(2).sum() +
                i.norm(dim=1).pow(2).sum() +
                j.norm(dim=1).pow(2).sum() +
                biasI.pow(2).sum() +
                biasJ.pow(2).sum()
        )

        return -log_prob + regularization

    def recommend(self, u):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]

        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.U[u, :]
        x_ui = torch.mm(u, self.V.t()) + self.biasV
        pred = torch.argsort(x_ui, dim=1)
        return pred
