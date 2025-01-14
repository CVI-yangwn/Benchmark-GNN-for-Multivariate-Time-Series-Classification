from yacs.config import CfgNode as CN
import yaml

_C = CN()

# 通用设置
_C.SYSTEM = CN()
_C.SYSTEM.GPU = 7
_C.SYSTEM.NUM_WORKERS = 10
_C.SYSTEM.SEED = 42

# 数据集设置
_C.DATASET = CN(new_allowed=True)
_C.DATASET.bands = None

# 实验设置
_C.EXPERIMENT = CN()
_C.EXPERIMENT.BATCH_SIZE = 128
_C.EXPERIMENT.EPOCHS = 200
_C.EXPERIMENT.OPTIMIZER = CN(new_allowed=True)
_C.EXPERIMENT.SCHEDULER = CN(new_allowed=True)

# 图构建
_C.GRAPH = CN()
_C.GRAPH.NODE = "raw"
_C.GRAPH.ADJ_MATRIX = "complete"

# 模型设置
_C.MODEL = CN(new_allowed=True)

def get_cfg_defaults() -> CN:
  return _C.clone()

def decompose(*args, **kargs):
  return

def merge_from_file_withoutsafe(_C: CN, cfg_filename: str) -> None:
    """Load a yaml config file and merge it this CfgNode."""
    with open(cfg_filename, "r") as f:
        cfg_as_dict = yaml.load(f.read(), yaml.FullLoader)
        cfg = CN(cfg_as_dict)
    _C.merge_from_other_cfg(cfg)

_VALID_TYPES = {tuple, list, str, int, float, bool}


def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

if __name__ == "__main__":
    cfg = _C.clone()
    # cfg.merge_from_file("config/raw_abspearson_GCN_ArticularyWordRecognition.yml")
    # cfg.MODEL.PARAM.dropout = 0.4
    try:
        print(cfg.fs)
    except AttributeError as e:
        print("None")