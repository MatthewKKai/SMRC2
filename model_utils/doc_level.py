import pandas as pd
import numpy as np
from torch.nn import Module

class doc_level(Module):
    def __init__(self, config):
        super(doc_level)
        self.config = config

    def attention(self):
        pass