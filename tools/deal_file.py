import os, shutil
import re, yaml
import pandas as pd
import numpy as np

generator = {}

def register_generator(name):
    def wrapper(func):
        generator[name] = func
    return wrapper

@register_generator("gcn")
def generate_gcn_yml(dataset, fs, band, adj, node, ncls, length):
    if node == "differential_entropy":
        indim = band
    elif node == "raw":
        indim = int(length)
    else:
        indim = int(length/2) 
    config = {
        "DATASET": {
            "CLASS": "UEADataset",
            "PARAM": {
                "name": dataset,
                "path": "./dataset/Multivariate2018_npz"
            },
            "bands": band,
            "fs": fs
        },
        "EXPERIMENT": {
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": {
                "CLASS": "SGD",
                "PARAM": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.0005
                }
            },
            "SCHEDULER": {
                "CLASS": "ReduceLROnPlateau",
                "PARAM": {
                    "cooldown": 0,
                    "eps": 8.0e-09,
                    "factor": 0.5,
                    "min_lr": 1.0e-06,
                    "mode": "min",
                    "patience": 10,
                    "threshold": 4.0e-05,
                    "threshold_mode": "rel",
                    "verbose": False
                }
            }
        },
        "GRAPH": {
            "ADJ_MATRIX": adj if adj != "diffgraphlearn" else "identity",
            "NODE": node
        },
        "MODEL": {
            "CLASS": "GCN",
            "PARAM": {
                "dropout": 0.1,
                "graphlearn": False if adj != "diffgraphlearn" else True,
                "hidden_dim": 128,
                "in_dim": indim,
                "len": length,
                "mlp_dim": 128,
                "n_classes": ncls,
                "n_layers": 3,
                "residual": True,
                "t_embedding": 128
            }
        },
        "SYSTEM": {
            "GPU": 7,
            "NUM_WORKERS": 10,
            "SEED": 42
        }
    }
    return config

@register_generator("megat")
def generate_megat_yml(dataset, fs, band, adj, node, ncls, length, thred=0.5):
    if node == "differential_entropy":
        indim = band
    elif node == "raw":
        indim = int(length)
    else:
        indim = int(length/2) 
    config = {
        "DATASET": {
            "CLASS": "UEADataset",
            "PARAM": {
                "name": dataset,
                "path": "./dataset/Multivariate2018_npz"
            },
            "bands": band,
            "fs": fs
        },
        "EXPERIMENT": {
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": {
                "CLASS": "SGD",
                "PARAM": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.0005
                }
            },
            "SCHEDULER": {
                "CLASS": "ReduceLROnPlateau",
                "PARAM": {
                    "cooldown": 0,
                    "eps": 8.0e-09,
                    "factor": 0.5,
                    "min_lr": 1.0e-06,
                    "mode": "min",
                    "patience": 10,
                    "threshold": 4.0e-05,
                    "threshold_mode": "rel",
                    "verbose": False
                }
            }
        },
        "GRAPH": {
            "ADJ_MATRIX": adj if adj != "diffgraphlearn" else "identity",
            "NODE": node
        },
        "MODEL": {
            "CLASS": "MEGAT",
            "PARAM": {
                "dropout": 0.1,
                "graphlearn": False if adj != "diffgraphlearn" else True,
                "hidden_dim": 128,
                "in_dim": indim,
                "len": length,
                "mlp_dim": 128,
                "n_classes": ncls,
                "n_layers": 3,
                "num_heads": 1,
                "readout": "mean",
                "residual": True,
                "t_embedding": 128,  # default:128
                "thred": thred
            }
        },
        "SYSTEM": {
            "GPU": 7,
            "NUM_WORKERS": 10,
            "SEED": 42
        }
    }
    return config


@register_generator("gat")
def generate_gat_yml(dataset, fs, band, adj, node, ncls, length):
    if node == "differential_entropy":
        indim = band
    elif node == "raw":
        indim = int(length)
    else:
        indim = int(length/2) 
    config = {
        "DATASET": {
            "CLASS": "UEADataset",
            "PARAM": {
                "name": dataset,
                "path": "./dataset/Multivariate2018_npz"
            },
            "bands": band,
            "fs": fs
        },
        "EXPERIMENT": {
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": {
                "CLASS": "SGD",
                "PARAM": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.0005
                }
            },
            "SCHEDULER": {
                "CLASS": "ReduceLROnPlateau",
                "PARAM": {
                    "cooldown": 0,
                    "eps": 8.0e-09,
                    "factor": 0.5,
                    "min_lr": 1.0e-06,
                    "mode": "min",
                    "patience": 10,
                    "threshold": 4.0e-05,
                    "threshold_mode": "rel",
                    "verbose": False
                }
            }
        },
        "GRAPH": {
            "ADJ_MATRIX": adj if adj!="diffgraphlearn" else "identity",
            "NODE": node
        },
        "MODEL": {
            "CLASS": "GAT",
            "PARAM": {
                "dropout": 0.1,
                "graphlearn": False  if adj != "diffgraphlearn" else True,
                "hidden_dim": 128,
                "in_dim": indim,
                "len": length,
                "mlp_dim": 128,
                "n_classes": ncls,
                "n_layers": 3,
                "num_heads": 1,
                "readout": "mean",
                "residual": True,
                "self_loop": True,
                "thred": 0.5
            }
        },
        "SYSTEM": {
            "GPU": 7,
            "NUM_WORKERS": 10,
            "SEED": 42
        }
    }
    return config

@register_generator("chebnet")
def generate_chebnet_yml(dataset, fs, band, adj, node, ncls, length):
    if node == "differential_entropy":
        indim = band
    elif node == "raw":
        indim = int(length)
    else:
        indim = int(length/2) 
    config = {
        "DATASET": {
            "CLASS": "UEADataset",
            "PARAM": {
                "name": dataset,
                "path": "./dataset/Multivariate2018_npz"
            },
            "bands":band,
            "fs": fs
        },
        "EXPERIMENT": {
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": {
                "CLASS": "SGD",
                "PARAM": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.0005
                }
            },
            "SCHEDULER": {
                "CLASS": "ReduceLROnPlateau",
                "PARAM": {
                    "cooldown": 0,
                    "eps": 8.0e-09,
                    "factor": 0.5,
                    "min_lr": 1.0e-06,
                    "mode": "min",
                    "patience": 10,
                    "threshold": 4.0e-05,
                    "threshold_mode": "rel",
                    "verbose": False
                }
            }
        },
        "GRAPH": {
            "ADJ_MATRIX": adj if adj != "diffgraphlearn" else "identity",
            "NODE": node
        },
        "MODEL": {
            "CLASS": "ChebNet",
            "PARAM": {
                "dropout": 0.1,
                "graphlearn": False if adj != "diffgraphlearn" else True,
                "hidden_dim": 128,
                "in_dim": indim,
                "k": 3,
                "len": length,
                "mlp_dim": 128,
                "n_classes": ncls,
                "n_layers": 3,
                "residual": True
            }
        },
        "SYSTEM": {
            "GPU": 7,
            "NUM_WORKERS": 10,
            "SEED": 42
        }
    }
    return config

@register_generator("stgcn")
def generate_stgcn_yml(dataset, fs, band, adj, node, ncls, length):
    if node == "differential_entropy":
        indim = band
    elif node == "raw":
        indim = int(length)
    else:
        indim = int(length/2) 
    config = {
        "DATASET": {
            "CLASS": "UEADataset",
            "PARAM": {
                "name": dataset,
                "path": "./dataset/Multivariate2018_npz"
            },
            "bands": band,
            "fs": fs
        },
        "EXPERIMENT": {
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": {
                "CLASS": "SGD",
                "PARAM": {
                    "lr": 0.001,
                    "momentum": 0.9,
                    "weight_decay": 0.0
                }
            },
            "SCHEDULER": {
                "CLASS": "ReduceLROnPlateau",
                "PARAM": {
                    "cooldown": 0,
                    "eps": 8.0e-09,
                    "factor": 0.5,
                    "min_lr": 1.0e-06,
                    "mode": "min",
                    "patience": 10,
                    "threshold": 4.0e-05,
                    "threshold_mode": "rel",
                    "verbose": False
                }
            }
        },
        "GRAPH": {
            "ADJ_MATRIX": adj if adj!="diffgraphlearn" else "identity",
            "NODE": node
        },
        "MODEL": {
            "CLASS": "STGCN",
            "PARAM": {
                "dropout": 0.1,
                "graphlearn": False if adj!="diffgraphlearn" else True,
                "hidden_dim": 128,
                "in_dim": indim,
                "k": 3,
                "len": length,
                "mlp_dim": 128,
                "n_classes": ncls,
                "n_layers": 3,
                "readout": "mean",
                "residual": True,
                "t_embedding": 128
            }
        },
        "SYSTEM": {
            "GPU": 7,
            "NUM_WORKERS": 10,
            "SEED": 42
        }
    }
    return config

def generate_yml():
    dst_dir = r"/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking/config/new"
    # datasets=("ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket",
    #           "DuckDuckGeese","EigenWorms","Epilepsy","EthanolConcentration",
    #           "ERing","FaceDetection","FingerMovements","HandMovementDirection",
    #           "Handwriting","Heartbeat","Libras","LSST","MotorImagery",
    #           "NATOPS","PenDigits","PEMS-SF","PhonemeSpectra","RacketSports",
    #           "SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary")
    # n_classes=(25,3,4,12,5,5,4,4,6,2,2,4,26,2,15,14,2,6,10,7,39,4,2,2,3,8)
    # fss=(200,128,10,184,0,0,16,0,0,250,100,0,0,0,0,0,1000,0,0,0,0,10,256,256,500,100)
    # lengths=(144,640,100,1197,270,17984,206,1751,65,62,50,400,152,405,45,36,3000,51,8,144,217,30,896,1152,2500,315)
    conf = [('ArticularyWordRecognition', 25, 200, 144), ('AtrialFibrillation', 3, 128, 640),
            ('BasicMotions', 4, 10, 100), ('Cricket', 12, 184, 1197),
            ('DuckDuckGeese', 5, 0, 270), ('EigenWorms', 5, 0, 17984),
            ('Epilepsy', 4, 16, 206), ('EthanolConcentration', 4, 0, 1751), 
            ('ERing', 6, 0, 65), ('FaceDetection', 2, 250, 62), ('FingerMovements', 2, 100, 50), 
            ('HandMovementDirection', 4, 0, 400), ('Handwriting', 26, 0, 152), ('Heartbeat', 2, 0, 405), 
            ('Libras', 15, 0, 45), ('LSST', 14, 0, 36), ('MotorImagery', 2, 1000, 3000), 
            ('NATOPS', 6, 0, 51), ('PenDigits', 10, 0, 8), ('PEMS-SF', 7, 0, 144), 
            ('PhonemeSpectra', 39, 0, 217), ('RacketSports', 4, 10, 30), 
            ('SelfRegulationSCP1', 2, 256, 896), ('SelfRegulationSCP2', 2, 256, 1152), 
            ('StandWalkJump', 3, 500, 2500), ('UWaveGestureLibrary', 8, 100, 315)]
    
    adjs=("mutual_information","abspearson","complete","diffgraphlearn")
    nodes=("raw","differential_entropy","power_spectral_density")
    gnns=("chebnet","gat","gcn","megat","stgcn")
    band = 5
    for dataset, ncls, fs, length in conf:
        for adj in adjs:
            for node in nodes:
                if node != "raw" and fs==0:
                    break
                for net in gnns:
                    if net == "megat" and adj == "diffgraphlearn":
                        cfg = generator[net](dataset, fs, band, adj, node, ncls, length, thred=0.0)
                    else:
                        cfg = generator[net](dataset, fs, band, adj, node, ncls, length)
                    os.makedirs(os.path.join(dst_dir, dataset), exist_ok=True)
                    yml_path = os.path.join(dst_dir, dataset, f"{net}-{adj}-{node}.yml")
                    with open(yml_path, 'w') as stream:
                        yaml.safe_dump(cfg, stream)



def change_yml_name():
    # ArticularyWordRecognition AtrialFibrillation FingerMovements HandMovementDirection Heartbeat
    dataset = "Heartbeat"
    new_yml_path = f"~/code/MTSC/MTSC-Graph-benchmarking/config/de/{dataset}"
    old_yml_path = f"~/code/MTSC/MTSC-Graph-benchmarking/config/old/{dataset}"

    if not os.path.exists(old_yml_path):
        raise ValueError("please check dataset name")
    os.makedirs(new_yml_path, exist_ok=True)

    def get_name(yml):
        nn, adj, node = None, None, None
        for n in ("chebnet", "megat", "stgcn"):  # gat megat gcn stgcn
            if n in yml: 
                nn = n 
                break
        if not nn:
            for n in ("gat", "gcn"):  # gat megat gcn stgcn
                if n in yml: 
                    nn = n 
                    break

        for a in ("mutual_information", "abspearson", "complete", "diffgraphlearn", "phase_locking_value"):
            if a in yml: 
                adj = a
                break

        for d in ("raw", "differential_entropy", "power_spectral_density"):
            if d in yml: 
                node = d
                break
        
        return nn, adj, node

    for yml in os.listdir(old_yml_path):
        nn, adj, node = get_name(yml)
        if nn and adj and node:
            new_yml_name = f"{nn}-{adj}-{node}.yml"
            print(new_yml_name)
            shutil.copy(os.path.join(old_yml_path, yml), os.path.join(new_yml_path, new_yml_name))

def get_best_acc(dataset, gnn, adj, node, seed):
    filePath = f"/data/yangwennuo/code/MTSC/logs/train/{dataset}/{gnn}-{adj}-{node}-{seed}.log"
    if not os.path.exists(filePath):
        return
    with open(filePath, 'r') as f:
        data = f.read()

    # Best accuray is 68.33333333333333 at eopch 197
    value_match = re.search(f"Best accuray is ([\d.]+) at eopch", data)
    if value_match:
        return "{:.3f}".format(float(value_match.group(1)))
    else:
        print(f"{dataset}/{gnn}-{adj}-{node}-{seed} read wrong")

def convert_all2csv():
    # datasets=("UWaveGestureLibrary","Cricket","Heartbeat")
    datasets=("ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket",
            "DuckDuckGeese","EigenWorms","Epilepsy","EthanolConcentration",
            "ERing","FaceDetection","FingerMovements","HandMovementDirection",
            "Handwriting","Heartbeat","Libras","LSST","MotorImagery",
            "NATOPS","PenDigits","PEMS-SF","PhonemeSpectra","RacketSports",
            "SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary")
    adjs=("mutual_information","abspearson","complete","diffgraphlearn")
    nodes=("raw","differential_entropy","power_spectral_density")
    gnns=("chebnet","gat","gcn","megat","stgcn")
    seeds=(42,152,310)
    with open("./train.txt", 'a') as f:
        for dataset in datasets:
            f.write(f"{dataset}\n")
            for gnn in gnns:
                f.write(f"{gnn}\n")
                t = ",,,".join(adjs)
                f.write(f" ,{t}\n")
                for node in nodes:
                    f.write(f"{node},")
                    for adj in adjs:
                        for seed in seeds:
                            acc = get_best_acc(dataset, gnn, adj, node, seed)
                            f.write(f"{acc},")
                    f.write("\n")
                f.write("\n")


def convert_max2csv():
    datasets=("ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket",
            "DuckDuckGeese","EigenWorms","Epilepsy","EthanolConcentration",
            "ERing","FaceDetection","FingerMovements","HandMovementDirection",
            "Handwriting","Heartbeat","Libras","LSST","MotorImagery",
            "NATOPS","PenDigits","PEMS-SF","PhonemeSpectra","RacketSports",
            "SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary")
    data_except = ("ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket",
            "Epilepsy",
            "FaceDetection","FingerMovements",
            "MotorImagery",
            "RacketSports",
            "SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary")
    adjs=("mutual_information","abspearson","complete","diffgraphlearn")
    nodes=("raw","differential_entropy","power_spectral_density")
    gnns=("chebnet","gat","gcn","megat","stgcn")
    seeds=(42,152,310)
    with open("./train.txt", 'a') as f:
        for dataset in datasets:
            f.write(f"{dataset}\n")
            for gnn in gnns:
                f.write(f"{gnn}\n")
                t = ",".join(adjs)
                f.write(f" ,{t}\n")
                for node in nodes:
                    if dataset not in data_except and node in ("differential_entropy","power_spectral_density"):
                        break
                    f.write(f"{node},")
                    for adj in adjs:
                        accs = []
                        for seed in seeds:
                            acc = get_best_acc(dataset, gnn, adj, node, seed)
                            try:
                                acc = float(acc)
                                accs.append(acc)
                            except:
                                pass
                        f.write("{:.3f},".format(np.max(np.array(accs)) if accs!=[] else 0))
                    f.write("\n")
                f.write("\n")

def convert_avg_dataset2csv():
    datasets=("ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket",
            "DuckDuckGeese","EigenWorms","Epilepsy","EthanolConcentration",
            "ERing","FaceDetection","FingerMovements","HandMovementDirection",
            "Handwriting","Heartbeat","Libras","LSST","MotorImagery",
            "NATOPS","PenDigits","PEMS-SF","PhonemeSpectra","RacketSports",
            "SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary")
    data_except = ("ArticularyWordRecognition","AtrialFibrillation","BasicMotions","Cricket",
            "Epilepsy",
            "FaceDetection","FingerMovements",
            "MotorImagery",
            "RacketSports",
            "SelfRegulationSCP1","SelfRegulationSCP2","StandWalkJump","UWaveGestureLibrary")
    adjs=("mutual_information","abspearson","complete","diffgraphlearn")
    nodes=("raw","differential_entropy","power_spectral_density")
    gnns=("chebnet","gat","gcn","megat","stgcn")
    seeds=(42,152,310)
    with open("./train.txt", 'a') as f:
        f.write(f',{",,,".join(gnns)}\n')
        f.write(",")
        for _ in range(5):
            f.write(f'{",".join(["raw","de","psd"])},') 
        f.write("\n")
        
        for adj in adjs:
            f.write(f"{adj},")
            for gnn in gnns:
                for node in nodes:
                    avg_acc = []
                    for dataset in datasets:
                        if dataset not in data_except:
                            continue  # next dataset
                        accs = []
                        for seed in seeds:
                            acc = get_best_acc(dataset, gnn, adj, node, seed)
                            try:
                                acc = float(acc)
                                accs.append(acc)
                            except:
                                pass
                        avg_acc.append(np.max(np.array(accs)))
                    f.write("{:.3f},".format(np.mean(np.array(avg_acc))))
            f.write("\n")
        f.write("\n")

        f.write(f',{",".join(gnns)}\n')
        f.write(",\n")
        for adj in adjs:
            f.write(f"{adj},")
            for gnn in gnns:
                node = 'raw'
                avg_acc = []
                for dataset in datasets:
                    if dataset in data_except:
                        continue  # next dataset
                    accs = []
                    for seed in seeds:
                        acc = get_best_acc(dataset, gnn, adj, node, seed)
                        try:
                            acc = float(acc)
                            accs.append(acc)
                        except:
                            pass
                    avg_acc.append(np.max(np.array(accs)))
                f.write("{:.3f},".format(np.mean(np.array(avg_acc))))
            f.write("\n")
        f.write("\n")

def cal_avg_for_each_type():

    HAR = ["BasicMotions", "Cricket", "Epilepsy", "RacketSports", "UWaveGestureLibrary"]
    MC = ["ArticularyWordRecognition"]
    ECG = ["AtrialFibrillation", "StandWalkJump"]
    EEG = ["FingerMovements", "MotorImagery", "SelfRegulationSCP1", "SelfRegulationSCP2", "FaceDetection"]
    ASC = []
    OTHER = []

    adjs=("mutual_information","abspearson","complete","diffgraphlearn")
    nodes=("raw","differential_entropy","power_spectral_density")
    gnns=("chebnet","gat","gcn","megat","stgcn")
    seeds=(42,152,310)

    def get_variable_name(variable):
        for name, value in locals().items():
            if value is variable:
                return name
        return None
    
    for datasets, name in zip([HAR, MC, ECG, EEG], ['HAR', 'MC', 'ECG', 'EEG']):
        
        with open(f"./{name}.txt", 'a') as f:
            f.write(f',{",,,".join(gnns)}\n')
            f.write(",")
            for _ in range(5):
                f.write(f'{",".join(["raw","de","psd"])},') 
            f.write("\n")
            
            for adj in adjs:
                f.write(f"{adj},")
                for gnn in gnns:
                    for node in nodes:
                        avg_acc = []
                        for dataset in datasets:
                            accs = []
                            for seed in seeds:
                                acc = get_best_acc(dataset, gnn, adj, node, seed)
                                try:
                                    acc = float(acc)
                                    accs.append(acc)
                                except:
                                    pass
                            avg_acc.append(np.max(np.array(accs)))
                        f.write("{:.3f},".format(np.mean(np.array(avg_acc))))
                f.write("\n")
            f.write("\n")




if __name__ == "__main__":
    # generate_yml()

    # convert_all2csv()
    # convert_max2csv()
    # convert_avg_dataset2csv()

    cal_avg_for_each_type()