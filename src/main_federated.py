from .preprocess.dataset import get_dataset
from .utils.create_clients import create_clients
from .client.client import Client
from .server.server import Server
from torch.utils.data import DataLoader


def main():

    train_set, test_set = get_dataset()

    num_clients = 100

    # creates the data shards for each client
    client_data = create_clients(train_set, num_clients)

    # creates Client instances and passes them the data shards for training
    clients = [Client(data, batch_size=8, lr=0.02, epochs=1) for data in client_data]

    # testing data for evaluation
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)
    
    # instance of server to perform the aggregation
    server = Server(clients, test_loader, num_rounds=4, lr=0.01, epochs=1)

    # loop to perform federated learning
    server.federated_averaging()
    

if __name__=="__main__":
    main()
