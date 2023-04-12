from torch.nn import Module
import scispacy, spacy
import transformers
from model_utils.graph_utils import *
from transformers import BertTokenizer, BertModel

# entities_library_path = r'./entities2ids'

class entity_level(Module):
    def __init__(self, Config):
        super(entity_level, self).__init__()
        self.config = Config
        self.nlp = spacy.load(Config.ner_corpus)
        self.pretrained_model = BertModel.from_pretrained(Config.version)
        self.tokenizer = BertTokenizer.from_pretrained(Config.version)
        self.gcn = nn.Sequential(
            GCN(768, Config.gcn_dim),
            GCN(Config.gcn_dim, 768)
        )
        self.fc = nn.Linear(768, 512)


    def forward(self, abs):
        entities = list(self.entity_extraction(abs))
        entity_tokens = self.tokenizer(entities, max_length=100, truncation=True, padding=True, return_tensors='pt')
        entity_initial_features = self.pretrained_model(**entity_tokens)
        out, features = self.gcn(entity_initial_features['pooler_output'])

        return self.fc(features)


    def entity_extraction(self, abs):
        return self.nlp(abs).ents

    # def create_graph(self, abs):
    #     entity_list = self.entity_extraction(abs)

