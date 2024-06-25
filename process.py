from config import *
import matplotlib.pyplot as plt


def count_text_len():
    """
    统计句子字长
    """
    text_len = []
    with open(TRAIN_SAMPLE_PATH, encoding="UTF-8") as f:
        for line in f.readlines():
            text, _ = line.split("\t")
            text_len.append(len(text))
    plt.hist(text_len)
    plt.show()
    print(max(text_len))

if __name__ == "__main__":
    count_text_len() # 38
    pass