import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re
import numpy as np
from .load_data import load_reviews
from .lstm_moviereviewdataset import MovieReviewDataset


# 1. Load Data and Labels
# Assuming 'load_reviews' gives you the lists 'reviews' and 'labels'
# Example:
# reviews = ["I love this movie", "This movie is terrible", ...]
# labels = [1, 0, ...]

reviews, labels = load_reviews  # Replace with actual loading function

# 2. Preprocessing: Tokenization and Vocabulary Building
def preprocess_text(text):
    """Cleans text: lowercase, removes special characters."""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower().strip())
    return text.split()

def encode_review(review, vocab_to_int):
        return [vocab_to_int.get(word, 0) for word in preprocess_text(review)]

def get_loaders(batch_size = 16):
    all_tokens = []
    for review in reviews:
        all_tokens.extend(preprocess_text(review))

    vocab = Counter(all_tokens)
    vocab_to_int = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}  # +1 for padding
    vocab_to_int["<PAD>"] = 0  # Padding token


    encoded_reviews = [encode_review(review, vocab_to_int) for review in reviews]
    labels = torch.tensor(labels, dtype=torch.float32)

    # Pad sequences to the same length
    padded_reviews = pad_sequence([torch.tensor(r) for r in encoded_reviews],
                                batch_first=True, padding_value=0)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(padded_reviews, labels, test_size=0.2, random_state=42)


    train_dataset = MovieReviewDataset(X_train, y_train)
    test_dataset = MovieReviewDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab_to_int


