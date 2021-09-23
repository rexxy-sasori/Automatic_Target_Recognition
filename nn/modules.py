"""
Author: Rex Geng

neural network modules utilized by models in models.py
"""

import torch
from torch import nn
from torch.nn import functional as F


class Squash(nn.Module):
    def __init__(self):
        super(Squash, self).__init__()

    def forward(self, x):
        """
        @param:
            x (batch_size, number of capsules, vec dim)

        return: squashed vectors
        """
        _, _, vec_dim = x.shape
        l2_norms = torch.norm(x, dim=-1)
        sq_l2_norms = l2_norms ** 2
        factors = sq_l2_norms / ((1 + sq_l2_norms) * l2_norms)
        factors = factors.unsqueeze(-1).repeat(1, 1, vec_dim)
        return factors * x


class PrimaryCapsule(nn.Linear):
    def __init__(self, input_dim, num_sar_cap):
        super(PrimaryCapsule, self).__init__(input_dim, input_dim * num_sar_cap, bias=False)
        self.num_sar_cap = num_sar_cap
        self.input_dim = input_dim

    def forward(self, x):
        """
        @param:
            x: (batch_size, input_dim)

        each projection matrix with in the capsule projects the input vector to TARGET vectors
        return: projected capsule output (batch_size, number of target capsule, input_dim)
        """
        batch_size, input_dim = x.shape
        out = F.linear(x, self.weight)
        out = out.reshape(batch_size, self.num_sar_cap, input_dim)
        return out


class PrimaryCapLayer(nn.Module):
    def __init__(self, num_primary_cap, num_sar_cap, input_dim):
        super(PrimaryCapLayer, self).__init__()
        self.num_primary_cap = num_primary_cap
        self.num_sar_cap = num_sar_cap
        self.input_dim = input_dim

        self.primary_capsules = []
        for _ in range(num_primary_cap):
            self.primary_capsules.append(PrimaryCapsule(input_dim, num_sar_cap))

        self.primary_capsules = nn.ModuleList(self.primary_capsules)

    def forward(self, x):
        """
        @param:
            x: (batch_size, input_dim)

        return (batch_size, num_sar_cap, num_primary_cap, input_dim)
        """
        primary_capsules = self.primary_capsules

        outs = []
        for idx, cap in enumerate(primary_capsules):
            out = cap(x[:, idx, :].squeeze(1))
            outs.append(out.unsqueeze(2))

        ret = torch.stack(outs, dim=2)
        ret = ret.squeeze(3)
        return ret


class VotingCapLayer(nn.Conv2d):
    def __init__(self, num_primary_cap, num_sar_cap, input_dim):
        super(VotingCapLayer, self).__init__(num_sar_cap, num_sar_cap,
                                             kernel_size=(num_primary_cap, 1),
                                             groups=num_sar_cap,
                                             bias=False)
        self.num_primary_cap = num_primary_cap
        self.num_sar_cap = num_sar_cap
        self.input_dim = input_dim

    def forward(self, x):
        """
        @param:
            x: (batch_size, num_sar_cap, num_primary_cap, input_dim)

        return voted tensor with shape (batch_size, num_sar_cap, input_dim)
        """
        out = F.conv2d(x, self.weight, groups=self.groups)
        out = out.squeeze(2)
        return out


class CapsuleClassifier(nn.Module):
    def __init__(self, num_primary_cap, num_sar_cap, input_dim):
        super(CapsuleClassifier, self).__init__()
        self.num_primary_cap = num_primary_cap
        self.num_sar_cap = num_sar_cap
        self.input_dim = input_dim

        self.squasher = Squash()
        self.prim_cap_layer = PrimaryCapLayer(num_primary_cap, num_sar_cap, input_dim)
        self.vote_cap = VotingCapLayer(num_primary_cap, num_sar_cap, input_dim)

    def forward(self, x):
        u = self.squasher(x)  # u_tilda to u (B, I, D) -> (B, I, D)
        u_hat = self.prim_cap_layer(u)  # u to u_hat (B, I, D) -> (B, J, I, D)
        s = self.vote_cap(u_hat)  # u_hat to s (B, J, I, D) -> (B, J, D)
        v = self.squasher(s)  # s to v (B, J, D) -> (B, J, D)
        v_norm = torch.norm(v, dim=-1)  # (B, J, D) -> (B, J)
        return v_norm


class CapsuleLayer(nn.Module):
    def __init__(self, num_in_channel, num_out_channel, kernel_size, stride, num_primary_cap, num_sar_cap, input_dim):
        super(CapsuleLayer, self).__init__()
        self.conv = nn.Conv2d(num_in_channel, num_out_channel, kernel_size=kernel_size, stride=stride)
        self.classifier = CapsuleClassifier(num_primary_cap, num_sar_cap, input_dim)
        self.num_primary_cap = num_primary_cap
        self.num_sar_cap = num_sar_cap
        self.input_dim = input_dim
        assert num_out_channel % self.input_dim == 0

    def forward(self, x):
        conv_out = self.conv(x)
        N, C, H, W = conv_out.shape
        conv_out = conv_out.squeeze(1)
        conv_out = conv_out.reshape(N, C // self.input_dim, self.input_dim, H, W)
        conv_out = conv_out.permute(0, 1, 3, 4, 2)
        conv_out = conv_out.reshape(N, -1, self.input_dim)
        u_tilda = conv_out
        v_norm = self.classifier(u_tilda)
        return v_norm
