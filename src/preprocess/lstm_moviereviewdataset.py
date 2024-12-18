import re
import string
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from collections import Counter

def cleantext(text):
    """Cleans the text by removing special characters, punctuation, stop words, and lemmatizing."""
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Lowercase the text
    text = text.lower()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels, vocab=None, max_len=100):
        self.max_len = max_len
        self.encoder = LabelEncoder()
        self.labels = torch.tensor(self.encoder.fit_transform(labels), dtype=torch.long)
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocab(reviews)
        else:
            self.vocab = vocab
            
        # Clean, tokenize and encode reviews
        self.reviews = [self.encode_review(cleantext(review)) for review in reviews]

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
