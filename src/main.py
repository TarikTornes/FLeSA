from .preprocess.load_data import load_reviews
from .preprocess.preprocessor import MovieReviewDataset
from .model.bertmodel import BERTMovieReviewClassifier, train_model, predict_review

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle, os


def main1():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    label_encoder = LabelEncoder()

    if os.path.isfile("./data/movie_review.pkl"):
        print("Pickled dataset is present. \nLoading pickle movie_review.pkl")
        with open("./data/movie_review.pkl", "rb") as f:
            data_pickle = pickle.load(f)

        reviews = data_pickle["reviews"]
        labels = data_pickle["labels"]
    else:
        print("Loading movie review data!")
        reviews, labels, _, _, _, _ = load_reviews("./data/movRev_data")
        pkl_dict = {"reviews": reviews, "labels": labels}

        with open("./data/movie_review.pkl", "wb") as f:
            pickle.dump(pkl_dict, f)
        print("Movie review data saved into movie_review.pkl")




    label_encoder = LabelEncoder()
    print(len(labels))
    print(set(labels))
    labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42
    )

    print("y_train: ", set(y_train))
    print("y_test: ",set(y_test))


    train_dataset = MovieReviewDataset(X_train, y_train, tokenizer)
    test_dataset = MovieReviewDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = BERTMovieReviewClassifier(num_classes=2)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    train_model(model, train_loader, test_loader, criterion, optimizer)

    torch.save(model.state_dict(), './data/bert_movie_review_classifier.pth')


def main2():
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTMovieReviewClassifier(num_classes=2)

    model.load_state_dict(torch.load('./data/bert_movie_review_classifier.pth'))

    if torch.backends.mps.is_available():
            device = "mps"
            print("Using device: mps")
    elif torch.backends.cuda.is_available():
        device = "cuda"
        print("Using device: cuda")
    else:
        device = "cpu"
        print("Using device: cpu")

    model.to(device)

    while True:
        
        rev = str(input())

        print("\n\n")
        print("Result: ", predict_review(rev, model, tokenizer, device))






if __name__=="__main__":
    main1()
