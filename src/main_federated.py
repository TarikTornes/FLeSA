from .preprocess.dataset import get_dataset
from .utils.create_clients import create_clients
from .client.client import Client
from .server.server import Server
from torch.utils.data import DataLoader


def main():

    train_set, test_set = get_dataset()

    num_clients = 4

    client_data = create_clients(train_set, num_clients)

    clients = [Client(data, batch_size=32, lr=0.01, epochs=1) for data in client_data]

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    server = Server(clients, test_loader, num_rounds=3, lr=0.01, epochs=1)

    server.federated_averaging()
    

if __name__=="__main__":
    main()
