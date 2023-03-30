from torch.nn import Module
import scispacy, spacy
import transformers
from transformers import AutoModel, AutoTokenizer

class entity_level(Module):
    def __init__(self, config):
        super(entity_level)
        self.config = config

    def entity_extraction(self, doc):
        return

    def graph_creation(self):
        pass

    def masking(self):
        pass