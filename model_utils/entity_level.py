from torch.nn import Module
import scispacy, spacy
import transformers
from transformers import AutoModel, AutoTokenizer

class entity_level(Module):
    def __init__(self, Config):
        super(entity_level, self).__init__()
        self.config = Config
        self.nlp = spacy.load(Config.ner_corpus)


    def forward(self):
        pass

    def entity_extraction(self, abs):
        return self.nlp(abs).ents

    def create_graph(self, abs):
        entity_list = self.entity_extraction(abs)


    def masking(self):
        pass