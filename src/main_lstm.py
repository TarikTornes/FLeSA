import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
from .preprocess.load_data import load_reviews
from .utils.check_device import get_device
from tqdm import tqdm

# 1. Preprocessing and Tokenization
class SentimentDataset(Dataset):
    def __init__(self, reviews, labels, vocab=None, max_len=100):
        self.max_len = max_len
        self.encoder = LabelEncoder()
        self.labels = torch.tensor(self.encoder.fit_transform(labels), dtype=torch.long)
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(reviews)
        else:
            self.vocab = vocab
            
        # Tokenize and encode reviews
        self.reviews = [self.encode_review(review) for review in reviews]

    def build_vocab(self, reviews):
        """Builds a vocabulary with a word-to-index mapping."""
        counter = Counter()
        for review in reviews:
            counter.update(review.split())
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.items())}  # Reserve 0, 1 for padding/unknown
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab
    
    def encode_review(self, review):
        """Encodes a review into a fixed-length sequence of word indices."""
        tokens = review.split()
        encoded = [self.vocab.get(word, 1) for word in tokens]  # 1 for <UNK>
        # Pad or truncate the sequence to max_len
        if len(encoded) < self.max_len:
            encoded += [0] * (self.max_len - len(encoded))  # Pad
        else:
            encoded = encoded[:self.max_len]  # Truncate
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx], self.labels[idx]

# 2. LSTM Model Definition
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

# 3. Training the Model
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

# 4. Evaluation Function
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

# 5. Main Function
def main():
    reviews, labels = load_reviews()
    # Hyperparameters
    embed_dim = 128
    hidden_dim = 64
    output_dim = 1
    num_layers = 2
    dropout = 0.5
    batch_size = 64
    epochs = 5
    max_len = 100
    learning_rate = 0.001
    device = get_device()
    
    # Split data into training and validation sets
    reviews_train, reviews_temp, labels_train, labels_temp = train_test_split(reviews, labels, test_size=0.3, random_state=42)
    reviews_val, reviews_test, labels_val, labels_test = train_test_split(reviews_temp, labels_temp, test_size=0.5, random_state=42)
    
    # Prepare datasets and data loaders
    train_dataset = SentimentDataset(reviews_train, labels_train, max_len=max_len)
    val_dataset = SentimentDataset(reviews_val, labels_val, vocab=train_dataset.vocab, max_len=max_len)
    test_dataset = SentimentDataset(reviews_test, labels_test, vocab=train_dataset.vocab, max_len=max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, criterion, and optimizer
    vocab_size = len(train_dataset.vocab)
    model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim, num_layers, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device)

    print("Training complete.")
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
