import numpy as np
import torch


def create_clients(dataset, num_clients=3):
    ''' This function splits the dataset into subsets for
        each client (i.e. shards)
        Code is from: https://techestate.io/demystifying-federated-machine-learning-a-practical-guide-with-pytorch/

        Args:
            dataset (PyTorch Dataset): whole dataset (features+labels)
            num_clients (int): number of clients

        Return:
            client_data (List[Dataset]): Data subset for each client
    '''

    data_len = len(dataset)
    indices = list(range(data_len))
    split_size = data_len // num_clients

    np.random.seed(42)
    np.random.shuffle(indices)

    client_data = []
    for i in range(num_clients):
        start = i * split_size
        end = start + split_size
        # Takes a subset split_size samples from indices to get
        # a shuffled dataset for a client of split_size
        client_data.append(torch.utils.data.Subset(dataset, indices[start:end]))

    return client_data
