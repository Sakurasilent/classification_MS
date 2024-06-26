

from torch.utils import data
from config import *
from transformers import BertTokenizer

import torch

from sklearn.metrics import classification_report

class Dataset(data.Dataset):
    def __init__(self,type="train") -> None:
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'dev':
            sample_path = DEV_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH
        self.lines = open(sample_path, encoding='utf-8').readlines()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        text, label = self.lines[item].split('\t')
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [0] * pad_len
            mask += [0] * pad_len
        target = int(label)
        return torch.tensor(input_ids[:TEXT_LEN]), \
               torch.tensor(mask[:TEXT_LEN]), \
               torch.tensor(target)
def get_label():
    text = open(LABEL_PATH, encoding='utf-8').read()
    # print(text)
    id2label = text.split()
    return id2label, {v: k for k, v in enumerate(id2label)}


def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )
def evaluate(pred, true, target_names=None, output_dict=False):
    return classification_report(
        true,
        pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )

if __name__ == "__main__":
    # lines = open(TEST_SAMPLE_PATH, encoding="utf-8").readlines()
    # print(lines)
    # text, label = lines[0].split("\t")
    # print(text)
    # print(label)
    # tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    # tokend = tokenizer(text)
    # print(tokend)
    
    # Dataset 测试 
    dataset = Dataset()
    loader = data.DataLoader(dataset=dataset, batch_size=2)
    # print(iter(loader))
    # print(loader.__iter__())
    for item in loader:
        print(item)
        break
    pass