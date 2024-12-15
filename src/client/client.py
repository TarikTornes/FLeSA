from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from ..model.bertmodel import BERTMovieReviewClassifier as Net
from ..utils.check_device import get_device
from ..utils.tracker import MemoryProfiler


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
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        ''' This function trains the model of the specific client object

        Return:
            return1 (state_dict): new learned weights of the model
        '''

        profiler = MemoryProfiler()

        profiler.log_memory("Start training")

        device = get_device()
        self.model.to(device)


        # torch.mps.empty_cache()
        # profiler.log_memory("Emptied cache")

        for epoch in range(self.epochs):
            self.model.train()
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
                
                # torch.mps.synchronize()

                if (counter+1) % 25 == 0:
                    print(f"Processing batch {counter+1}/{len(self.train_loader)}")

        
        profiler.log_memory("End training")

        torch.mps.empty_cache()

        return self.model.state_dict()

