# Analysis of Attention Mechanism in Time Series Forecasting

In this repo, you can find our implementation and experimental results of several modifications to the attention mechanism. 

## Create environment
If you wish to use a GPU, the poetry install might differ. 
- ```conda env create -f environment.yml -y```
- ```conda activate ou-dne-transformers```
- ```poetry install```
- ```pre-commit install```

## Traffic Dataset

We used the [Traffic dataset](https://archive.ics.uci.edu/dataset/204/pems+sf). The preprocessed dataset can be found on [OSF](https://osf.io/qe6jp/). Copy OSF data to `/data/processed/`

