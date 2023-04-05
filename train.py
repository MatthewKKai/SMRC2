from model_utils.text_level import text_encoder
import config
from torch import optim
from torch.utils.data import DataLoader
from data_loader import create_dataset
import os

Config = config.get_opt()

def train():
    # params
    params = {
        'batch_size': Config.batch_size,
        'shuffle': True
    }
    epoch = 100

    # prepare data
    dataset = create_dataset('./data.json')
    dataloader = DataLoader(dataset, **params)

    # load models
    textEncoder = text_encoder(Config)

    # training
    for abs, label in dataloader:
        # print(abs)
        text_embedding = textEncoder()
        print(label)
        print('------------')






    optimizer = optim.AdamW()

    # clear gradients
    optimizer.zero_grad()

if __name__=="__main__":
    train()