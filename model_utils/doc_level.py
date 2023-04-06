import pandas as pd
import numpy as np
from torch.nn import Module

class doc_level(Module):
    def __init__(self, Config):
        super(doc_level, self).__init__()
        self.config = Config
        pass


    def attention(self):
        pass


    def masking(self):
        pass