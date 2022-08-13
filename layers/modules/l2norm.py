import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    '''
    The size of the conv4_3 feature map is 38x38, the network layer is at the front, the norm is large,
    and an L2 Normalization needs to be added to ensure that the difference with the subsequent detection layer is not very large
    '''
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        # Convert a non-trainable type Tensor to a trainable type parameter
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    # Initialization parameters
    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        # Calculate the 2-norm of x
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        # Expand the dimension of self.weight to shape[1,512,1,1], and then calculate with reference to the formula
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
