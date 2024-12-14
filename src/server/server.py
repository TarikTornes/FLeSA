from ..model.bertmodel import BERTMovieReviewClassifier as Net
from ..model.bertmodel import evaluate
import copy

import torch

class Server:
    """
    A class to represent the server of the FL model

    Attr:
        clients(List[clients]): Represents the clients within the FL system
        test_loader(DataLoader): test data in order to evaluate the model
        num_rounds(int): number of rounds the fl model will train/request
                        clients training
        lr(float): Learning rate for training the central model
        epochs(int): Number of epochs for training central/global model
        global_model(nn.Model): Global model to be trained
    """

    def __init__(self, clients, test_loader, num_rounds, lr, epochs):
        self.clients = clients
        self.test_loader = test_loader
        self.num_rounds = num_rounds
        self.lr = lr
        self.epochs = epochs
        self.global_model = Net()


    def init_train():
        """
        Initilize the global model with a small dataset in 
        order to have a staring point.
        """

        pass


    def federated_averaging(self):
        """
        Performs the global federated learning procedure
        """

        for round in range(self.num_rounds):
            client_weights = []

            for client in self.clients:

                # represents sending the global weights to the client
                client.model.load_state_dict(self.global_model.state_dict())
                
                # represents the global training of the client
                client_weights.append(client.train())


            # updates weights of global model to the avg of client weights
            self.global_model.load_state_dict(self.average_weights(client_weights))

            # computes the accuracy of the global model
            accuracy = evaluate(self.global_model, self.test_loader)

            print(f'Round {round} | Accuracy: {accuracy:.4f}')

        print("Completed Federated Averaging!")


    def average_weights(self, client_weights):
        """
        Averages/Aggregates the weigths of the different clients

        Args:
            client_weights (List[weights]): list of clients weights

        Return:
            avg_weights (weights): new weights for the global model
        """

        avg_weights = copy.deepcopy(client_weights[0])

        num_clients = len(client_weights)

        for w in avg_weights.keys():

            weight_sum = torch.zeros_like(avg_weights[w])

            for client_weight in client_weights:
                weight_sum += client_weight[w]

            avg_weights[w] = weight_sum / num_clients

        return avg_weights







