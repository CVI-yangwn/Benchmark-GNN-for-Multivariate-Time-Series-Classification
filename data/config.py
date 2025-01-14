import collections

class GlobalDict(collections.UserDict):
    def __missing__(self, key):
        return None

datasets = GlobalDict()

def register_dataset(model):
    datasets[model.__name__] = model
    return model

if __name__ == "__main__":
    print(datasets)