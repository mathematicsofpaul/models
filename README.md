# Oscer Project Test
Presented here is an ensemble of three different RNN architectures developed for the task of NER classification. In particular, the RNNs used were vanilla RNNs, GRU RNNs, and LSTM RNNs. The language used to develop the code was PyTorch. 

I highly recommend starting with the Guide notebook just to get a feel for what is going on! 

## Setup/Installation

This project was built and tested with Python 3.8. The required packages and their versions are located in the requirements.txt file. 

To run this project, first clone the repository and install the required python packages with the requirements.txt:

```
cd <directory you want to install to>
git clone https://github.com/mathematicsofpaul/models
cd models
conda create -n paul_nguyen python=3.8 
pip install -r requirements.txt 

#pip install ipykernel #if you intend on experimenting with the notebook 
#pip install notebook 
```

The following are the neccessary commands to run the code: 

```
python train.py 
python test.py
python pipeline.py "Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia." 
```
Note: the phrase above was drawn from a random article from the internet. (out of sample)  

## Things you should know: 

1. Since I used Glove embeddings for the vocab vector assignment, the initial waiting time to download the embeddings was quite long roughly ~7min or so. 
2. The analysis was done on an Nvidia gpu 1080ti with CUDA 11.0. 
3. I have set the N_EPOCH value to 40, but if it takes too long to train then the value can be adjusted in the config.py file.    


