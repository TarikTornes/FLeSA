import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np
from ..preprocess.lstm_preprocess import encode_review

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])  # Use the last layer's hidden state
        return self.sigmoid(out)
    

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        reviews, labels = batch
        reviews, labels = reviews.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(reviews).squeeze()
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

# 7. Evaluate the Model
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            reviews, labels = batch
            reviews, labels = reviews.to(device), labels.to(device)
            predictions = model(reviews).squeeze()
            preds = (predictions > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


# Optional: Load and Predict
def predict_sentiment(model, text, vocab_to_int, device):
    model.eval()
    encoded = encode_review(text, vocab_to_int)
    padded = torch.tensor(encoded).unsqueeze(0).to(device)
    prediction = model(padded)
    return "Positive" if prediction.item() > 0.5 else "Negative"

# Example:
# print(predict_sentiment(model, "The movie was fantastic!", vocab_to_int, device))