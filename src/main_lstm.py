import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .preprocess.load_data import load_reviews
from .preprocess.lstm_moviereviewdataset import MovieReviewDataset
from .utils.check_device import get_device
from .model.lstmmodel import *



# 5. Main Function
def main():
    reviews, labels = load_reviews()
    # Hyperparameters
    embed_dim = 256
    hidden_dim = 256
    output_dim = 1
    num_layers = 2
    dropout = 0.5
    batch_size = 64
    epochs = 10
    max_len = 100
    learning_rate = 0.001
    device = get_device()
    
    # Split data into training and validation sets
    reviews_train, reviews_temp, labels_train, labels_temp = train_test_split(reviews, labels, test_size=0.3, random_state=42)
    reviews_val, reviews_test, labels_val, labels_test = train_test_split(reviews_temp, labels_temp, test_size=0.5, random_state=42)
    
    # Prepare datasets and data loaders
    train_dataset = MovieReviewDataset(reviews_train, labels_train, max_len=max_len)
    val_dataset = MovieReviewDataset(reviews_val, labels_val, vocab=train_dataset.vocab, max_len=max_len)
    test_dataset = MovieReviewDataset(reviews_test, labels_test, vocab=train_dataset.vocab, max_len=max_len)
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
