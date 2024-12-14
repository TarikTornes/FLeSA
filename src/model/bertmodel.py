from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import BertModel
from ..utils.check_device import get_device

class BERTMovieReviewClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super(BERTMovieReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # gets the embeddings from the pretrained bert model
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
        )
        
        # pooler output, represents the content of the 
        # whole sequence (BERT specific [CLS] token, which can be
        # seen as a summary)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits


def evaluate(model, test_loader):
    """
    Evaluates the model in order to compute
    its performance.

    Args:
        model (nn.Model): model to be evaluated
        test_loader (DataLoader): Test data for evaluation

    Return:
        accuracy (float): accuracy score of the model on the test set
    """
    device = get_device()
    model.to(device)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    return accuracy




def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3):
    """
    Trains a model on given hyperparameters. And prints the performance on the 
    validation set.

    Args:
        model (nn.Model): model to be trained
        train_loader (DataLoader): data on which the model will be trained
        val_loader (DataLoader): data on which we will perform validation after training
        criterion (Loss function): criterion to compute the loss
        optimizer (Optimizer): optimization algorithm
        epochs (int): number of epoch the model will train
    """

    
    device = get_device()
    model.to(device)


    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Training)', leave=False)

        counter = 0
        for batch in train_loader:
            counter += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_train_loss += loss.item()
            train_progress.update(1)

            train_progress.set_postfix({'Loss': loss.item()})

            # print(f"Processing batch {counter+1}/{len(train_loader)}")


        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0 

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Validation)', leave=False)

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)

                loss = criterion(outputs, labels)

                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                val_progress.set_postfix({'Loss': loss.item()})

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {total_train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {total_val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {100 * correct_predictions/total_predictions:.2f}%')




def predict_review(review, model, tokenizer, device):
    """
    Predicts sentiment of a review with a trained model

    Args:
        review (str): text representing a movie review
        model (nn.Model): trained model which will perform the prediction
        tokenizer (Tokenizer): tokenizer corresponding to the training data
        device: device on which the model should evaluate

    Return:
        return1 (str): "Positive"/"Negative" predicted by the model
    """
    model.eval()
    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        _, predicted = torch.max(outputs, 1)

    return 'Positve' if predicted.item() == 1 else 'Negative'



