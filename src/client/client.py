from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from ..model.bertmodel import BertModel as Net


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
        self.optimizer = optim.SGD(self.model.parameters, lr=lr)
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        ''' This function trains the model of the specific client object

        Return:
            return1 (state_dict): new learned weights of the model
        '''

        for _ in range(self.epochs):

            for _, (data, target) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()

