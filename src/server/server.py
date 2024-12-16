from ..model.bertmodel import BERTMovieReviewClassifier as Net
from ..model.bertmodel import evaluate
from ..utils.tracker import MemoryProfiler, trace_mem
import copy
from pympler import asizeof
import random
import tracemalloc
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
        round_config_fn (Callable): is a function that defines which clients
                                    should be taken in a specific round
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


    def round_config_fn(self, round):

        if round == 0:
            num_clients = 3
        else:
            num_clients = 2


        return random.sample(self.clients, num_clients)



    def federated_averaging(self):
        """
        Performs the global federated learning procedure
        """

        tracemalloc.start()

        trace_mem("Before round loop")


        for round in range(self.num_rounds):
            print(f"\n--- Round {round} ---")
            trace_mem("in round")

            client_weights = []

            # checks if config function is available
            if self.round_config_fn:
                # sets config for the current round
                round_clients = self.round_config_fn(round)
            else:
                round_clients = self.clients

            for client in round_clients:

                # represents sending the global weights to the client
                client.model.load_state_dict(self.global_model.state_dict())

                trace_mem("After sending global weights")
                w1 = client.train()
                trace_mem("After training")
                
                # represents the global training of the client
                client_weights.append(w1)
                trace_mem("After appending weights")




            # updates weights of global model to the avg of client weights
            self.global_model.load_state_dict(self.average_weights(client_weights))

            # trace_mem("After averaging weights")

            # torch.mps.synchronize()

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

        # avg_weights = copy.deepcopy(client_weights[0])
        avg_weights = client_weights[0]

        num_clients = len(client_weights)

        for w in avg_weights.keys():

            weight_sum = torch.zeros_like(avg_weights[w])

            for client_weight in client_weights:
                weight_sum += client_weight[w]

            avg_weights[w] = weight_sum / num_clients



        return avg_weights









