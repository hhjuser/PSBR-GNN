import torch
from torch.nn import Linear
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_p):
        super().__init__()
        # torch.manual_seed(12345)
        self.p = dropout_p
        hidden_channels = 16
        self.lin1 = Linear(num_features, hidden_channels)
        # self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, num_classes)
        self.is_dense = False # actually doesn't matter

    def forward(self, x,*argv):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p = self.p, training=self.training)
        # x = self.lin2(x)
        # x = x.relu()
        # x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin3(x)
        # x = Q @ x
        return F.log_softmax(x, dim=1)