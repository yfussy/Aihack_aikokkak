BASE_DIR = "../../"


# get train/test dataset -> getDataDir("train"/"test") [w/o version]
# get cleaned train/test dataset -> getDataDir("train"/"test", version) [w/ version]
def getDataDir(type: str, version: int=0) -> str:
    dir = BASE_DIR + "data/"
    if (version):
        dir += "cleaned_"
    if (type == "train"):
        dir += "train_dataset"
    if (type == "test"):
        dir += "test_without_gt"
    if (version):
        dir += "_v" + str(version)
    return dir + ".csv"

# get model -> getModelDir("model"/"scaler"/"feature", version) [w/o custom]
# get model w/ custom name -> getModeldir("name", version, True) [w/ custom]
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

# get prediction -> getPredDir(version) [w/o name]
# get prediction w/ custom name -> getPredDir(version, "name") [w/ name]
def getPredDir(version: int, name: str="") -> str:
    if (name):
        return BASE_DIR + "prediction/" + name + "_v" + str(version) + ".csv"
    return BASE_DIR + "prediction/predictions_v" + str(version) + ".csv"