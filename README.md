# FL_SentimentAnalysis
This repository demonstrates a small example of using Federated Learning (FL) for Sentiment Analysis.
This project was created by Thuc K. N. and Tarik T. in order to get familiar with Federated Learning
and compare the advantages and disadvantages of compared to a classical centralized learning approach.

For this project the Movie Review Dataset ![from https://ai.stanford.edu/~amaas/data/sentiment/] was used.

## Pre-requisites

### Software Environment
1. Install depedencies listed in environment.yaml
2. Change configs to your preferences in `conf/config.yml`

### Installation
#### 1. Clone Repository
Clone the repo to your local machine
``` shell
# Using HTTPS
git clone https://github.com/TarikTornes/FL_SentimentAnalysis.git

# Using SSH
git clone git@github.com:TarikTornes/FL_SentimentAnalysis.git
```

#### 2. Install depedencies

Pip:
``` shell
pip install -r requirements.txt
```

Conda:
``` shell
conda env create --file environment.yaml
```

#### 3. Dataset installation
Put the directory which can be downloaded into the `/data/` and rename it to `/data/movRev_data`.
Such that the structure of the data directory will look the following:

```
data/
└─ movRev_data/
    ├── README
    ├── imdb.vocab
    ├── imdbEr.txt
    ├── test/
    └── train/
```

### Running the code
The general idea of this repo is to first implement and then train two "classical"
central models, i.e. one basing on the LSTM architecture and one basing on the Transformer architecture using BERT, on sentiment analysis.
The next step is the implementation and training of a federated learning model for the same task using the BERT model for the client training in order to get familiar with the federated learning and demonstrate the concept.

There are 3 main files that can be run:
1. *main_bert.py*:
This predicts or trains (depending on the configuration) the sentiment analysis model which bases on the Transformer encoder.

2. *main_lstm.py*:
This predicts or trains (depending on the configuration) the sentiment analysis model which bases on the LSTM architecture.

3. *main_federated.py*:
This simulates the federated learning procedure within an adjustable environment. One can set the number of clients, as well as the sample strategy for each round.

To run one of those files you need to be in the root directory of this repo.

For example to run the federated learning simulation, execute:
```shell
python3 -m src.main_federated
```

### Results
Without further hyperparameter finetuning we achieved an accuracy of 89.04% and an accuracy of 
xx.xx% for the LSTM model.
Due to some hardware limitations we needed to limit the number of rounds and clients that were training.
With 9 Rounds and sampling 4 out of 100 clients for the first round and 3 clients for the remaining rounds we have achieved an accuracy of 84.248%. The constant improvement of the model can be seen in the following graph, with the exception of an outlier.

![ressources/diagram_acc_fl.png]


### References
- Dataset: ![https://ai.stanford.edu/~amaas/data/sentiment/]
