import torch
from torch.utils import data
import os
import numpy as np
import config


class Mydata(data.Dataset):
    def __init__(self, root, corpus_name, stop_file, windows=2, neg_sam=5):
        super(Mydata, self).__init__()
        # 语料库文件
        self.data_path = os.path.join(root, corpus_name)
        # Skip窗口大小
        self.windows = windows
        # 每个词的负采样数量
        self.neg_sams = neg_sam * 2 * windows
        # 中文停用词文件
        self.stop_file = os.path.join(root, stop_file)
        # -, -, 编码后的序列, 词频表
        self.word2id, self.id2word, self.sequence, self.word2count = self.get_data()
        # 负采样频率
        word_freq = np.array([count for count in self.word2count.values()], dtype=np.float32)
        word_freq = word_freq**0.75 / np.sum(word_freq**0.75)
        self.word_freq = torch.tensor(word_freq)

    def __getitem__(self, index):
        center = self.sequence[index]
        # 周围词
        pos_index = list(range(index-self.windows, index)) + list(range(index+1, index+1+self.windows))
        pos_index = [i%len(self.sequence) for i in pos_index]
        pos_words = self.sequence[pos_index]
        # 返回负采样词
        neg_words = torch.multinomial(self.word_freq, self.neg_sams, False)
        # 数据放入device
        center = center.to(config.device)
        pos_words = pos_words.to(config.device)
        neg_words = neg_words.to(config.device)
        return center, pos_words, neg_words

    def __len__(self):
        return len(self.sequence)

    def get_data(self):
        # 词表字典
        word2id = {}
        id2word = {}
        # 词频率
        word2count = {}
        # 词表编码后的语料
        sequence = []
        with open(self.stop_file, 'r', encoding='utf-8') as file:
            stopwords = file.read().split()
        with open(self.data_path, 'r', encoding='utf-8') as file:
            words = file.read().split()
        print("original corpus size: ", len(words))
        vocal = [word for word in words if word not in stopwords]
        print("new corpus size: ", len(vocal))

        for word in vocal:
            if word not in word2id:
                index = len(word2id)
                word2id[word] = index
                id2word[index] = word
            word2count[word] = word2count.get(word, 0) + 1
            sequence.append(word2id[word])
        # print("size: ", len(word2id), len(id2word), len(word2count))
        sequence = torch.tensor(sequence)
        return word2id, id2word, sequence, word2count


def get_dataloader(root, corpus_name, stop_file, batch_size, shuffle=True, windows=2, neg_sam=5):
    dataset = Mydata(root=root,
                     corpus_name=corpus_name,
                     stop_file=stop_file,
                     windows=windows,
                     neg_sam=neg_sam)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset


# if __name__ == '__main__':
#     dataset = Mydata("./data", "zh.txt", "stopwords.txt")
#     print("-----------------")
#     print(dataset.__len__())