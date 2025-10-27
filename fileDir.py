BASE_DIR = "../../"

def getDataDir(type: str, version: int=0) -> str:
    dir = BASE_DIR + "data/"
    if (type == "cleaned"):
        return dir + "cleaned_data_v" + str(version) + ".csv"
    if (type == "train"):
        return dir + "train_dataset.csv"
    if (type == "test"):
        return dir + "test_without_gt.csv"

def getModelDir(type: str, version: int, custom: bool=False) -> str:
    dir = BASE_DIR + "model/v" + str(version) + "/"
    if (custom):
        return dir + type + ".pkl"
    if (type == "model"):
        return dir + "model.pkl"
    if (type == "scaler"):
        return dir + "scaler_model.pkl"
    if (type == "feature"):
        return dir + "train_features_model.pkl"

def getPredDir(version: int, name: str="") -> str:
    if (name):
        return BASE_DIR + "prediction/" + name + "_v" + str(version) + ".csv"
    return BASE_DIR + "prediction/predictions_v" + str(version) + ".csv"