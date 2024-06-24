"""
基于bert实现 实现 textcnn
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import torch
from config import *

class TextCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.bert = AutoModelForMaskedLM(BERT_MODEL)
        # bert不参与参数更新
        for name, param in self.bert.name_parameters():
            param.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM))for i in FILTER_SIZES])
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)
        


if __name__== "__main__":
    
    pass


