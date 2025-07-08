import torch
import torch.nn as nn
import re

# Simple tokenizer: splits by space and removes punctuation
def simple_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().split()

# Build vocab manually
texts = ["AI is changing education", "Teachers use technology to enhance learning"]
tokens = [token for line in texts for token in simple_tokenizer(line)]
vocab = {word: idx for idx, word in enumerate(set(tokens))}
vocab_size = len(vocab)

def encode(text):
    return torch.tensor([vocab.get(word, 0) for word in simple_tokenizer(text)], dtype=torch.long)

# LSTM model definition
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Sample run
if __name__ == "__main__":
    model = LSTMGenerator(vocab_size)
    model.eval()

    input_text = "AI is"
    encoded = encode(input_text).unsqueeze(0)
    output, _ = model(encoded)

    print("🔹 LSTM Output Tensor:")
    print(output)
