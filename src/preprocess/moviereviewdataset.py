import torch
from torch.utils.data import Dataset


class MovieReviewDataset(Dataset):
    """
    This class creates a PyTorch Dataset object in order
    to create a DataLoader afterwards.
    
    Args:
        reviews (List[str]): list of reviews
        labels (List[int]): list of labels for the reviews 1 or 0
        tokenizer : Transforms the review strings into tokens for the model
        max_len (int): Defines the maximum size of one review sample -> input size
    """
    def __init__(self, reviews, labels, tokenizer, max_len=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        inputs = self.tokenizer.encode_plus(
                review,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True
        )

        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
