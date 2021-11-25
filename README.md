# Oscer_project_test
Presented here is an ensemble of three different RNN architectures developed for the task of NER classification. In particular, the RNNs used were vanilla RNNs, GRU RNNs, and LSTM RNNs. The language used to develop the code was PyTorch. 

## Setup/Installation

This project was built and tested with Python 3.8. The required packages and their versions are located in the requirements.txt file. 

To run this project, first clone the repository and install the required python packages with the requirements.txt:

```
$ cd <directory you want to install to>
$ git clone https://github.com/mathematicsofpaul/models
$ cd models
$ pip install -r requirements.txt 
$ pip install notebook # if you intend on experimenting with the algorithm 
```

The following are the neccessary commands to run the code: 

```
$ python train.py 
$ python test.py
$ python pipeline.py "Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia." 

```
## Things you should know: 

1. Since I used Glove embeddings for the vocab vector assignment, the initial waiting time to download the embeddings were quite long roughly ~7min or so. 
2. The analysis was done on an Nvidia gpu 1080ti with CUDA 11.0. If you had the chance to, please run the code with the gpu that you have - preferably on CUDA 11.0 or so.
3. I have set the N_EPOCH value to 40, but if it takes too long to train then the value can be adjusted in the config.py file.    


