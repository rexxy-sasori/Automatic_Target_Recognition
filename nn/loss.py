import torch
from torch import nn
import torch.nn.functional as F

class CapMarginLoss(nn.modules.loss._Loss):
    def __init__(self, lam=0.5, num_class=10):
        super(CapMarginLoss, self).__init__()
        self.lam = lam
        self.num_class = num_class

    def forward(self, input, target):
        """
        @param:
            input: (batch_size, number of target capsules)
            target: (batch_size, number of target capsules)
        """
        m_plus = 0.9
        m_minus = 0.1
        
        target = F.one_hot(target, num_classes=self.num_class)
        #first_term = torch.max(torch.zeros_like(m_plus - input), m_plus - input) ** 2
        #second_term = torch.max(torch.zeros_like(input - m_minus), input - m_minus) ** 2
                
        first_term = torch.clamp(m_plus - input, min=0)**2
        second_term = torch.clamp(input - m_minus, min=0)**2
        
        #print("first",first_term.shape)
        #print("second",second_term.shape)
        #print("target",target.shape)
        #print("input",input.shape)
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
