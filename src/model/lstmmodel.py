import torch
import torch.nn as nn
from tqdm import tqdm


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last hidden state
        x = self.fc1(self.dropout(lstm_out))
        x = self.relu(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, device='cpu'):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for reviews, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            reviews, labels = reviews.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(reviews).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

def evaluate_model(model, val_loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for reviews, labels in val_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            outputs = model(reviews).squeeze()
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return avg_loss, accuracy

def test_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            outputs = model(reviews).squeeze()
            predictions = (outputs >= 0.5).float()
            correct += (predictions == labels).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")