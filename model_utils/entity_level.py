from torch.nn import Module
import scispacy, spacy
import transformers
from model_utils.graph_utils import *
from transformers import BertTokenizer, BertModel

# entities_library_path = r'./entities2ids'

class entity_encoder(Module):
    def __init__(self, Config):
        super(entity_encoder, self).__init__()
        self.config = Config
        self.nlp = spacy.load(Config.ner_corpus)
        self.pretrained_model = BertModel.from_pretrained(Config.version)
        self.tokenizer = BertTokenizer.from_pretrained(Config.version)
        self.gcn = GCN(768, 768, 'relu')
        self.fc = nn.Linear(768, 512)


    def forward(self, abs):
        entities = ''
        for ent in self.entity_extraction(abs):
            entities = entities + ' ' + str(ent)
        # print(entities)
        entity_tokens = self.tokenizer(entities, max_length=100, truncation=True, padding=True, return_tensors='pt')
        # print(len(entity_tokens))
        entity_initial_features = self.pretrained_model(**entity_tokens)
        # print("shape of entity_features: {}".format(entity_initial_features['pooler_output'].shape))
        out, features = self.gcn(entity_initial_features['pooler_output'], torch.randn(1, 1))

        return self.fc(features)


    def entity_extraction(self, abs):
        return self.nlp(abs).ents

    # def create_graph(self, abs):
    #     entity_list = self.entity_extraction(abs)

