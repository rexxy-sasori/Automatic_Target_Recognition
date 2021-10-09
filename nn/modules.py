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
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
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


class FeatureRefine(nn.Module):
    def __init__(self, input_channel, pool_dim):
        super(FeatureRefine, self).__init__()
        self.max_pool = nn.MaxPool2d(pool_dim)
        self.avg_pool = nn.AvgPool2d(pool_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_channel, out_features=input_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=input_channel, out_features=input_channel)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.shape
        c_ap = self.max_pool(x)
        c_xp = self.avg_pool(x)

        c_ap = c_ap.reshape(N, C)
        c_xp = c_xp.reshape(N, C)

        mlp_ap = self.mlp(c_ap)
        mlp_xp = self.mlp(c_xp)
        C = self.sigmoid(mlp_ap + mlp_xp)
        C = C.unsqueeze(-1).unsqueeze(-1)

        x_c = x * C


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset
