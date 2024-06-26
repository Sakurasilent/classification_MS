BERT_MODEL = "./hgmodel/bert-base-chinese/"

EMBEDDING_DIM = 768
NUM_FILTERS = 256
NUM_CLASSES = 10
FILTER_SIZES = [2, 3, 4]

TEXT_LEN = 35

TRAIN_SAMPLE_PATH = './data/train.txt'
DEV_SAMPLE_PATH = './data/dev.txt'
TEST_SAMPLE_PATH = './data/test.txt'

LABEL_PATH = './data/class.txt' #标签
LR= 1e-4
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR="./models_out/"

