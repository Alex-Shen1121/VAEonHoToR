import torch
import torch.nn as nn
import math


class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay):
        super().__init__()
        self.U = nn.Parameter(torch.empty(user_size, dim))
        self.V = nn.Parameter(torch.empty(item_size, dim))
        self.biasV = nn.Parameter(torch.empty(item_size))
        nn.init.xavier_normal_(self.U.data)
        nn.init.xavier_normal_(self.V.data)
        nn.init.normal_(self.biasV, 1e-3)
        self.weight_decay = weight_decay

    def forward(self, u, i, j):
        """Return loss value.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor
        """
        biasI = self.biasV[i]
        biasJ = self.biasV[j]
        u = self.U[u, :]
        i = self.V[i, :]
        j = self.V[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        x_uij = x_ui - x_uj
        # log_prob = F.logsigmoid(x_uij).sum()
        log_prob = x_uij.sigmoid().log().sum()
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
