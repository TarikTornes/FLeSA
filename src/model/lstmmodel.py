from tqdm import tqdm
import torch
import torch.nn as nn
from ..utils.check_device import get_device

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_dim  # Fix: Assign to self.hidden_size
        self.num_layers = num_layers  # Fix: Assign to self.num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Use the last time step
        return self.fc(out)
    


# Function to evaluate the model
def evaluate(model, test_loader):
    """
    Evaluates the model to compute its performance.

    Args:
        model (nn.Module): Model to be evaluated.
        test_loader (DataLoader): Test data for evaluation.

    Returns:
        accuracy (float): Accuracy score of the model on the test set.
    """
    device = get_device()
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3):
    """
    Trains the model and evaluates it on the validation set.

    Args:
        model (nn.Module): Model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (Loss function): Loss function.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of epochs for training.
    """
    device = get_device()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Training)', leave=False)

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress.update(1)
            train_progress.set_postfix({'Loss': loss.item()})

        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Validation)', leave=False)

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

                val_progress.set_postfix({'Loss': loss.item()})

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {total_train_loss / len(train_loader):.4f}')
        print(f'Val Loss: {total_val_loss / len(val_loader):.4f}')
        print(f'Val Accuracy: {100 * correct_predictions / total_predictions:.2f}%')

# Function to predict sentiment for a review
def predict_review(review, model, tokenizer):
    """
    Predicts the sentiment of a review using a trained model.

    Args:
        review (str): Text representing the review.
        model (nn.Module): Trained model.
        tokenizer: Tokenizer corresponding to the training data.

    Returns:
        prediction (str): "Positive" or "Negative" based on model prediction.
    """
    device = get_device()
    model.to(device)
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

    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)

    return 'Positive' if predicted.item() == 1 else 'Negative'