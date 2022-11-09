import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 100
epochs = 150
batch_size = 64
windows = 3
neg_sam = 5
root_dir = './data'
result_dir = './result'
en_corpus = 'en.txt'
zh_corpus = 'zh.txt'
stopwords = 'stopwords.txt'
lr = 0.01
