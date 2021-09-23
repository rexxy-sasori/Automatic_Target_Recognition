import torch
from torch import nn


class CapMarginLoss(nn._Loss):
    def __init__(self, lam=0.5):
        super(CapMarginLoss, self).__init__()
        self.lam = lam

    def forward(self, input, target):
        """
        @param:
            input: (batch_size, number of target capsules)
            target: (batch_size, number of target capsules)
        """
        m_plus = 0.9
        m_minus = 0.1
        first_term = torch.max(0, m_plus - input) ** 2
        second_term = torch.max(0, input - m_minus) ** 2

        target_complement = 1 - target
        loss = target * first_term + self.lam * target_complement * second_term
        loss = loss.sum(1)
        avg_loss = loss.mean()
        return avg_loss


__LOSS__ = {
    'cel': nn.CrossEntropyLoss,
    'wcel': nn.CrossEntropyLoss,
    'mse': nn.MSELoss,
    'cap_margin': CapMarginLoss,
}
