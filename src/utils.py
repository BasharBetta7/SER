from imports import * 

class CrossEntropyLoss(nn.Module):
    def __init__(self, type='normal'):
        super().__init__()
        self.gamma = 0.5
        self.type = type
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        if self.type == 'focal':
            loss = (1.0 - inputs[targets]) ** self.gamma * ce
            loss = loss.mean()
        else:
            loss = ce.mean()