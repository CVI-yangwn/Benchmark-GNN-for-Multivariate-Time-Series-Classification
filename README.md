# Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification
The code for the paper: Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification

## Usage

## Environment
Our platforms are Ubuntu 20.04.1 and Ubuntu 16.04.4. The detailed requirement of environment please see at file [environment.yml](./environment.yml). 

To set up the environment easily, you can run commands:
```shell
conda env create -f environment.yml -n <environment_name>
conda activate <environment_name>
```

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

**Parameter Description**

The following command-line arguments are supported by the script:

| Argument      | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `--log_path`  | Path to save the training log file.                                         |
| `--conf_file` | Path to the configuration file for model and training settings.             |
| `--save_path` | Directory to save the trained models.                                       |
| `--gpu`       | ID of the GPU device to use for training.                                   |
| `--seed`      | Random seed for reproducibility of experiments.                             |

---

**Usage Examples**

1. Training with Custom Parameters

```shell
python main.py --log_path $logFile --conf_file ./config/new/ERing/gcn-raw-complete.yml --save_path ./models --gpu 0 --seed 42
```

2. Training with a Script

```shell
bash scripts/train.sh
```
The [train.sh](./scripts/train.sh) script contains predefined configurations and can be modified to suit your needs.

### Test Command
After training the model, you can evaluate its performance using the provided test script.
```shell
bash scripts/test.sh
```

## Experiments

Due to the ongoing review process of our paper, the experimental details and results are currently not publicly available. We will update this section once the paper is accepted and published.

## Citation 

**Please kindly cite our papers if you used or were inspired by our idea:**

```
@article{yang2025benchmarking,
  title={Benchmarking Graph Representations and Graph Neural Networks for Multivariate Time Series Classification},
  author={Yang, Wennuo and Wu, Shiling and Zhou, Yuzhi and Luo, Cheng and He, Xilin and Xie, Weicheng and Shen, Linlin and Song, Siyang},
  journal={arXiv preprint arXiv:2501.08305},
  year={2025}
}
```

