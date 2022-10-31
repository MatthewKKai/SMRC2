import dgl
from torch import nn
from torch.nn import Module

class GCN(Module):
    def __init__(self, config):
        super(GCN)
        self.config = config

    def msg_receive(self):
        pass

    def msg_processing(self):
        pass

    def forward(self):
        pass
        # return node_representation, edge_representation

class Attention_Gate(nn.Module):
    def __init__(self, config):
        super(Attention_Gate, self).__init__()
        self.config = config

