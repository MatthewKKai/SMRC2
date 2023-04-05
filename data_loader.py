from torch.utils.data import Dataset
import json


class create_dataset(Dataset):
    def __init__(self, data_path):
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            print("Please provide correct data path or correct data form")




    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        abs = self.data[index]['abs']
        label = self.data[index]['label']

        return abs, label



