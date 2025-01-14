import collections

class GlobalDict(collections.UserDict):
    def __missing__(self, key):
        return None

models = GlobalDict()

def register_model(model):
    models[model.__name__] = model
    return model

if __name__ == "__main__":
    print(models)