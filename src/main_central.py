from .model.bertmodel import BERTMovieReviewClassifier, train_model, predict_review
from .preprocess.dataset import get_dataset
from .utils.check_device import get_device

from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main_train():

    # create pyTorch Dataset instances for the dataloader
    train_dataset, test_dataset = get_dataset()

    # create PyTorch DataLoader for the model
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # create instance of BERT model
    model = BERTMovieReviewClassifier(num_classes=2)
    # define criterion for training
    criterion = nn.CrossEntropyLoss()

    # define optimizer for training
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # train the model
    train_model(model, train_loader, test_loader, criterion, optimizer)

    #save the trained parameters to file
    torch.save(model.state_dict(), './data/bert_movie_review_classifier.pth')


def main_predict():
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTMovieReviewClassifier(num_classes=2)

    model.load_state_dict(torch.load('./data/bert_movie_review_classifier.pth'))

    device = get_device()

    model.to(device)

    while True:
        
        rev = str(input())

        print("\n\n")
        print("Result: ", predict_review(rev, model, tokenizer, device))






if __name__=="__main__":
    main_predict()
