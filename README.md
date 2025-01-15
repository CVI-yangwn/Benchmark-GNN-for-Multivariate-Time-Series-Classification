# Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification
The code for the paper: Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification

## Usage

## Environment
Our platforms are Ubuntu 20.04.1 and Ubuntu 16.04.4. The detailed requirement of environment please see at file [environment.yml](./environment.yml).


## Data Preparation
Download all of the new 30 multivariate UEA Time Series Classification datasets from [here](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip). \

Then unzip the file in direction MTSC-Graph-benchmarking/dataset/Multivariate2018_npz as the path tree shows
```
MTSC-Graph-benchmarking
   |- dataset
   |   |- Multivariate2018_npz
   |   |    |- ArticularyWordRecognition
   |   |    |- ...                      
```

### Train Command
```shell
python main.py --log_path $logFile --conf_file ./config/new/ERing/gcn-raw-complete.yml --save_path ./models --gpu 0 --seed 42
```
or
```shell
bash scripts/train.sh
```
### Test Command
```shell
bash scripts/test.sh
```

## Experiments

Due to the ongoing review process of our paper, the experimental details and results are currently not publicly available. We will update this section once the paper is accepted and published.

