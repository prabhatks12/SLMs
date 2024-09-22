import torch
import torch.nn as nn
import torch.nn.functional as F

vocabulary_len = 336

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        
    def forward(self, index, targets=None):
        embedded = self.embedding(index) # self.dropout(self.embedding(index))
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(self.dropout(lstm_out))
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]  # Focus on the last token
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# Initialize the model with higher dimensions
model = BigramLanguageModel(vocab_size=vocabulary_len, embedding_dim=256, hidden_dim=1024, num_layers=3, dropout=0.3)
