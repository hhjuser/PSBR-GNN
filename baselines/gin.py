import torch
from torch_geometric.nn import GINConv
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_p):
        super().__init__()
        self.p = dropout_p
        # torch.manual_seed(1234567)
        self.conv1 = GINConv(num_features, hidden_channels, add_self_loops=False)  #normalize=False
        self.conv2 = GINConv(hidden_channels, num_classes, add_self_loops=False)
        self.is_dense = False


    def forward(self, x, edge_index, edge_weight=None):
        edge_index = edge_index.long()
        edge_index = edge_index.coalesce().indices()
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)