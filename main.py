import torch
import config
import numpy as np
from mydata import get_dataloader
from model import SkipGram
import os
import json
import tqdm


def train(model, dataloader, learning_rate, epochs, save_name):
    model.train()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
    loss_list = []
    for epoch in tqdm.tqdm(range(epochs)):
        total_loss = 0
        for i, (center, pos_words, neg_words) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = model(center, pos_words, neg_words).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_list.append(total_loss)

    torch.save(model.state_dict(), os.path.join(config.result_dir, save_name + '_model.pth'))
    return loss_list


os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
### 选择语料类型: 'zh' | 'en' ###
corpus = 'en'
################################
corpus_name = config.zh_corpus if corpus == 'zh' else config.en_corpus
dataloader, dataset = get_dataloader(root=config.root_dir,
                                     corpus_name=corpus_name,
                                     stop_file=config.stopwords,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     windows=config.windows,
                                     neg_sam=config.neg_sam)
vocab_size = len(dataset.word2id)
model = SkipGram(vocab_size, config.embedding_dim).to(config.device)
loss_list = train(model=model, dataloader=dataloader, learning_rate=config.lr, epochs=config.epochs, save_name=corpus)
np.savetxt(corpus + "_loss.csv", np.array(loss_list), delimiter=',')

# 保存词向量
embedding_weights = model.get_weight()
dic = {word: embedding_weights[idx].tolist() for word, idx in dataset.word2id.items()}

with open(os.path.join(config.result_dir, corpus + '_embed.json'), 'w', encoding='utf-8') as file:
    file.write(json.dumps(dic, ensure_ascii=False, indent=4))
