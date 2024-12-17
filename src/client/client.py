from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from ..model.bertmodel import BERTMovieReviewClassifier as Net
from ..utils.check_device import get_device


class Client:
    ''' Represent a client
        
        Attr:
            train_loader (Torch DataLOader): contains the training data
            model:  Contains the local model
            optimizer: Contains the optimizer used for the training
            epochs: Amount of epochs to train
            criterion: Criterion used for the training
    '''

    def __init__(self, train_data, batch_size, lr, epochs):
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.model = Net()
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def train(self):
        ''' This function trains the model of the specific client object

        Return:
            return1 (state_dict): new learned weights of the model
        '''


        device = get_device()
        self.model.to(device)


        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.model.train()


        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')

            counter = 0
            for batch in self.train_loader:

                counter += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                self.optimizer.zero_grad()
                output = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                

                if (counter+1) % 25 == 0:
                    print(f"Processing batch {counter+1}/{len(self.train_loader)}")

        res = self.model.state_dict()

        return res

