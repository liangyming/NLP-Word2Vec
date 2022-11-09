import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import spatial
import json
import seaborn as sns
import adjustText
import random


def plot_loss(loss, name):
    length = len(loss)
    x = np.arange(1, length + 1)
    plt.plot(x, loss)
    plt.title(name)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


class Tool():
    def __init__(self, embedding_path):
        # 使得pyplot可输出汉字
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        file = open(embedding_path, 'r', encoding='utf-8')
        self.wordVec = json.load(file)

    def find_near_word(self, word, num):
        '''
        :param word: 选定的词
        :param num: 需要查找的近义词数量
        :return: num个近义词列表
        '''
        embedding = self.wordVec[word]
        cos_sim_list = []
        for key, value in self.wordVec.items():
            cos_sim = 1 - spatial.distance.cosine(value, embedding)
            cos_sim_list.append((cos_sim, key))
        cos_sim_list.sort(reverse=True)
        return cos_sim_list[0:num]

    def draw_heatmap(self, words):
        '''
        :param words: 一列的相近的词
        '''
        vectors = [self.wordVec[word] for word in words]
        f, ax = plt.subplots(figsize=(15, 9))
        sns.heatmap(vectors, ax=ax)
        ax.set_yticklabels(words)
        plt.show()

    def draw_scatter(self, words):
        '''
        :param words: 同上
        '''
        pca = PCA(n_components=2)
        vectors = [self.wordVec[word] for word in words]
        coordinates = pca.fit_transform(vectors)
        plt.figure(figsize=(13, 9))
        plt.scatter(coordinates[:, 0], coordinates[:, 1])
        text = [plt.text(coordinates[i, 0], coordinates[i, 1], words[i], fontsize=15) for i in range(len(words))]
        adjustText.adjust_text(text)
        plt.show()


if __name__ == '__main__':
    tool = Tool("./result/en_embed.json")
    near_list = tool.find_near_word('my', 9)
    words = [value for key, value in near_list]
    tool.draw_heatmap(words)

