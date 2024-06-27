"""
预测代码
"""

from config import *
from utils import *
from model import *

class Predict():
    def __init__(self) -> None:
        self.model = torch.load(MODEL_DIR_END , map_location=DEVICE)
        self.model = self.model.to(DEVICE)
        self.id2label,_ = get_label()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


    def predict(self,text:str):
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']
        if len(input_ids) < TEXT_LEN:
            pad_len = (TEXT_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len
        

        input_ids= torch.tensor([input_ids[:TEXT_LEN]]).to(DEVICE)
        mask = torch.tensor([mask[:TEXT_LEN]]).to(DEVICE)
        pred = self.model(input_ids, mask)
        pred_ = torch.argmax(pred, dim=1)
        return self.id2label[pred_]

if __name__ == "__main__":
    text = "2009年成人高考招生统一考试时间表"
    pre = Predict()
    result = pre.predict(text=text)
    print(result)
    pass