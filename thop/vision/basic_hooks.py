import logging

import torch
from torch.nn.modules.conv import _ConvNd

multiply_adds = 1


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = total_params


def zero_ops(m, x, y):
    m.total_ops += torch.DoubleTensor([int(0)])
    m.num_act = torch.DoubleTensor([int(0)])
    m.num_dp = torch.DoubleTensor([int(0)])

    m.input_reuse = torch.DoubleTensor([0])
    m.weight_reuse = torch.DoubleTensor([0])
    m.dim_dp = torch.DoubleTensor([0])


def count_convNd(m: _ConvNd, x: (torch.Tensor,), y: torch.Tensor):
    x = x[0]

    kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh

    bias_ops = 1 if m.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops)
    # print(y.size())
    # print(y.nelement())

    m.total_ops = torch.DoubleTensor([int(total_ops)])

    num_act = x.nelement()
    num_dp = y.nelement()
    m.num_act = torch.DoubleTensor([num_act])
    m.num_dp = torch.DoubleTensor([num_dp])
    m.dim_dp = torch.DoubleTensor([m.in_channels // m.groups * kernel_ops])

    m.input_reuse = torch.DoubleTensor([x.size()[2::].numel()])
    m.weight_reuse = torch.DoubleTensor([kernel_ops * y.size()[-3]])




def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    if not m.training:
        # subtract, divide, gamma, beta
        total_ops = 2 * nelements

    m.total_ops += torch.DoubleTensor([int(0)])
    m.num_act = torch.DoubleTensor([int(0)])
    m.num_dp = torch.DoubleTensor([int(0)])

    m.input_reuse = torch.DoubleTensor([0])
    m.weight_reuse = torch.DoubleTensor([0])
    m.dim_dp = torch.DoubleTensor([0])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()

    m.total_ops += torch.DoubleTensor([int(nelements)])
    m.num_act = torch.DoubleTensor([int(0)])
    m.num_dp = torch.DoubleTensor([int(0)])
    m.input_reuse = torch.DoubleTensor([0])
    m.weight_reuse = torch.DoubleTensor([0])
    m.dim_dp = torch.DoubleTensor([0])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    m.num_act = torch.DoubleTensor([int(0)])
    m.num_dp = torch.DoubleTensor([int(0)])
    m.input_reuse = torch.DoubleTensor([0])
    m.weight_reuse = torch.DoubleTensor([0])
    m.dim_dp = torch.DoubleTensor([0])


def count_avgpool(m, x, y):
    # total_add = torch.prod(torch.Tensor([m.kernel_size]))
    # total_div = 1
    # kernel_ops = total_add + total_div
    kernel_ops = 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(0)])
    m.num_act = torch.DoubleTensor([int(0)])
    m.num_dp = torch.DoubleTensor([int(0)])
    m.input_reuse = torch.DoubleTensor([0])
    m.weight_reuse = torch.DoubleTensor([0])
    m.dim_dp = torch.DoubleTensor([0])


def count_adap_avgpool(m, x, y):
    kernel = torch.DoubleTensor([*(x[0].shape[2:])]) // torch.DoubleTensor([*(y.shape[2:])])
    total_add = torch.prod(kernel)
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.DoubleTensor([int(0)])
    m.num_act = torch.DoubleTensor([int(0)])
    m.num_dp = torch.DoubleTensor([int(0)])
    m.input_reuse = torch.DoubleTensor([0])
    m.weight_reuse = torch.DoubleTensor([0])
    m.dim_dp = torch.DoubleTensor([0])


# TODO: verify the accuracy
def count_upsample(m, x, y):
    if m.mode not in ("nearest", "linear", "bilinear", "bicubic",):  # "trilinear"
        logging.warning("mode %s is not implemented yet, take it a zero op" % m.mode)
        return zero_ops(m, x, y)

    if m.mode == "nearest":
        return zero_ops(m, x, y)

    x = x[0]
    if m.mode == "linear":
        total_ops = y.nelement() * 5  # 2 muls + 3 add
    elif m.mode == "bilinear":
        # https://en.wikipedia.org/wiki/Bilinear_interpolation
        total_ops = y.nelement() * 11  # 6 muls + 5 adds
    elif m.mode == "bicubic":
        # https://en.wikipedia.org/wiki/Bicubic_interpolation
        # Product matrix [4x4] x [4x4] x [4x4]
        ops_solve_A = 224  # 128 muls + 96 adds
        ops_solve_p = 35  # 16 muls + 12 adds + 4 muls + 3 adds
        total_ops = y.nelement() * (ops_solve_A + ops_solve_p)
    elif m.mode == "trilinear":
        # https://en.wikipedia.org/wiki/Trilinear_interpolation
        # can viewed as 2 bilinear + 1 linear
        total_ops = y.nelement() * (13 * 2 + 5)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


# nn.Linear
def count_linear(m, x, y):
    x = x[0]
    # per output element
    total_mul = m.in_features
    # total_add = m.in_features - 1
    # total_add += 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops = total_mul * num_elements

    m.total_ops += torch.DoubleTensor([int(total_ops)])
    num_act = x.nelement()
    num_dp = y.nelement()
    m.num_act = torch.DoubleTensor([num_act])
    m.num_dp = torch.DoubleTensor([num_dp])

    m.input_reuse = torch.DoubleTensor([num_elements])
    m.weight_reuse = torch.DoubleTensor([0])

    m.dim_dp = torch.DoubleTensor([total_mul])
