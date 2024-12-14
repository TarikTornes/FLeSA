from .load_data import load_reviews
from .moviereviewdataset import MovieReviewDataset

from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

def get_dataset():
    '''This function will give us the dataset in a form that
        can be passed through the model.

        Return:
            trainset: training set implemented with torch Dataset
            testset: test set implemented with torch Dataset
    '''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()


    reviews, labels = load_reviews()
    labels = label_encoder.fit_transform(labels)


    X_train, X_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42
    )

    trainset = MovieReviewDataset(X_train, y_train, tokenizer)
    testset = MovieReviewDataset(X_test, y_test, tokenizer)

    # trainset= DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    # testset = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return trainset, testset

