from torch import nn
from torch.nn import Module
import torch
import config
from transformers import BertModel, BertTokenizer

Config = config.get_opt()

# convert doc into sentence-pairs
class text_encoder(Module):
    def __init__(self, Config):
        super(text_encoder, self).__init__()
        self.Config = Config
        self.tokenizer = BertTokenizer.from_pretrained(Config.version)
        self.pretrain_model = BertModel.from_pretrained(Config.version)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=Config.d_transformer, nhead=Config.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=Config.num_layers)
        self.linear = nn.Linear(768, 512)

    def forward(self, text):
        # get embedding
        inputs = self.tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors='pt')
        initial_embedding = self.pretrain_model(**inputs)
        initial_embedding_transit = self.linear(initial_embedding['pooler_output'])
        out = self.transformer_encoder(initial_embedding_transit)

        return out



