"""
基于bert实现 实现 textcnn
"""

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from config import *

class TextCNN(nn.Module):
    def __init__(self) -> None:
        super(TextCNN, self).__init__()
        
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        # bert不参与参数更新
        # for name, param in self.bert.name_parameters():
        #     param.requires_grad = False
        self.convs = nn.ModuleList([nn.Conv2d(1, NUM_FILTERS, (i, EMBEDDING_DIM))for i in FILTER_SIZES])
        self.linear = nn.Linear(NUM_FILTERS * 3, NUM_CLASSES)
    def conv_and_pool(self,conv,input):
        out = conv(input)
        out = F.relu(out)
        # squeeze去掉维度为一的位置
        # unsqueeze指定位置加一个维度
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input, mask):
        out = self.bert(input, mask)[0].unsqueeze(1)
        out = torch.cat([self.conv_and_pool(conv, out) for conv in self.convs], dim=1)
        return self.linear(out)


if __name__== "__main__":
    
    model = TextCNN()
    print(model)
    
    input = torch.randint(0, 3000, (2, TEXT_LEN))
    mask = torch.ones_like(input)
    print(input.shape)
    print(model(input, mask).shape)
    pass


