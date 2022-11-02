from torch import nn
from torch.nn import Module
from transformers import BertModel


# convert doc into sentence-pairs
class text_level(Module):
    def __init__(self, config):
        self.config = config
        self.biobert = BertModel.from_pretrained(r"") # add parameter later
        pass

    def pair_sampling(self):
        pass

