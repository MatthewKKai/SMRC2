from torch import nn
from torch.nn import Module
import torch
import config
from transformers import BertModel

Config = config.get_opt()

# convert doc into sentence-pairs
class text_encoder(Module):
    def __init__(self, Config):
        self.Config = Config
        self.biobert = BertModel.from_pretrained(r"") # add parameter later
        self.transformer = nn.Transformer(self.Config.d_transformer, batch_first=True)

    def forward(self):
        pass



