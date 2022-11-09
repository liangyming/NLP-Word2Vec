import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.out_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, pos_words, neg_words):
        input_embedding = self.in_embedding(center) # [batch_size, embedding_dim]
        pos_embedding = self.out_embedding(pos_words) # [batch_size, windows*2, embedding_dim]
        neg_embedding = self.out_embedding(neg_words) # [batch_size, windows*2*neg_num, embedding_dim]
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embedding_dim, 1]
        # unsqueeze()增加维度，suqueeze()降低维度
        pos_loss = torch.bmm(pos_embedding, input_embedding).squeeze() # [batch_size, window*2, 1]
        neg_loss = torch.bmm(neg_embedding, -input_embedding).squeeze() # [batch_size, window*2*num, 1]
        pos_loss = F.logsigmoid(pos_loss).sum(1)
        neg_loss = F.logsigmoid(neg_loss).sum(1)
        loss = pos_loss + neg_loss
        return -loss

    def get_weight(self):
        return self.in_embedding.weight.data.cpu().numpy()