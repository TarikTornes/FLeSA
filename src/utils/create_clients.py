import random

def create_clients(X, y, num_clients=3, initial='clients'):

    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    data = list(zip(X,y))
    random.shuffle(data)

    # determines the length of shard for each client
    size = len(data)//num_clients

    # splits the data into shards for each client
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    assert(len(shards) == len(client_names))

    # associates each client with a shard
    res = {client_names[i] : shards[i] for i in range(len(client_names))}

    return res
