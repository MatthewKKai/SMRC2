from model_utils.text_level import text_encoder
import config
from torch import optim
from torch import nn
import torch
from torch.utils.data import DataLoader
from data_loader import create_dataset
import os

Config = config.get_opt()

# label -> id
labels2ids = {'cardiovascular diseases': 0,
'chronic kidney disease': 1,
'chronic respiratory diseases': 2,
'ciabetes mellitus': 3,
'cigestive diseases': 4,
'hiv/aids': 5,
'hepatitis a/b/c/e': 6,
'mental disorders': 7,
'musculoskeletal disorders': 8,
'neoplasms (cancer)': 9,
'neurological disorders': 10}

label_onehot = nn.functional.one_hot(torch.tensor([0,1,2,3,4,5,6,7,8,9,10]), num_classes=11)

def train():
    # params
    params = {
        'batch_size': Config.batch_size,
        'shuffle': True
    }
    epoch = 10

    # prepare data
    dataset = create_dataset('./data.json')
    dataloader = DataLoader(dataset, **params)

    # load models
    textEncoder = text_encoder(Config)

    # create optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(textEncoder.parameters(), lr=Config.learning_rate)

    # training
    for i in range(epoch):
        total_loss = 0.0
        print('----------Epoch {}---------'.format(i))
        for abs, label in dataloader:
            optimizer.zero_grad()
            out = textEncoder(abs)
            # print(out.shape)
            label_id = torch.tensor([labels2ids[label[0]]])
            # print(label_id)
            loss = criterion(out, label_id)
            loss.backward()
            optimizer.step()

            # print
            total_loss += loss.item()
            print('total_loss is: {}'.format(total_loss))

    print("Finished")


if __name__=="__main__":
    train()