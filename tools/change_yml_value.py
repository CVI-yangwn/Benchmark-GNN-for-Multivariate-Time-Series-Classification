import yaml
import os
import sys

def freeze_param():
    keys, values = [], []

    keys.append(["DATASET", "PARAM", "path"])
    values.append("./dataset/Multivariate2018_npz")

    keys.append(["EXPERIMENT", "EPOCHS"])
    values.append(200)

    keys.append(["EXPERIMENT", "OPTIMIZER", "PARAM", "lr"])
    values.append(0.001)

    keys.append(["MODEL", "PARAM", "dropout"])
    values.append(0.1)

    keys.append(["EXPERIMENT", "BATCH_SIZE"])
    values.append(64)

    keys.append(["MODEL", "PARAM", "hidden_dim"])
    values.append(128)

    keys.append(["EXPERIMENT", "SCHEDULER", "PARAM", "patience"])
    values.append(10)

    return keys, values


def change_value(yml_path, keys, value):
    print(yml_path)
    with open(yml_path, "r") as f:
        cfg = yaml.load(f.read(), yaml.FullLoader)

    t :dict = cfg
    for k in keys[:-1]:
        t = t[k]
    if keys[-1] in t.keys():
        t[keys[-1]] = value
    
    with open(yml_path, 'w') as stream:
        yaml.safe_dump(cfg, stream)

if __name__ == "__main__":

    dataset = sys.argv[1]
    de_dim = int(sys.argv[2])

    # ArticularyWordRecognition AtrialFibrillation FingerMovements HandMovementDirection Heartbeat
    # dataset = "ArticularyWordRecognition"
    new_yml_dir = f"/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking/config/de/{dataset}"

    keys, values = freeze_param()

    # de_dim = 5
    keys.append(["DATASET", "bands"])
    values.append(de_dim)
    keys.append(["MODEL", "PARAM", "in_dim"])
    values.append(de_dim)

    for yml in os.listdir(new_yml_dir):
        yml_path = os.path.join(new_yml_dir, yml)

        for key, value in zip(keys, values):
            change_value(yml_path, key, value)
