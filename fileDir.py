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

# get type -> getModelDir("model"/"scaler"/"feature", version) [w/o model]
# get type w/ custom model -> getModeldir("model"/"scaler"/"feature", version, "model") [w/ model]
def getModelDir(type: str, version: int, model: str="") -> str:
    dir = BASE_DIR + "model/v" + str(version) + "/"
    if (type == "model"):
        dir += "model"
    if (type == "scaler"):
        dir += "scaler_model"
    if (type == "feature"):
        dir += "train_features_model"
    if (type == "param"):
        dir += "param_log"
    if (model):
        dir += "_" + model

    if (type == "param"):
        return dir + ".json"
    return dir + ".pkl"

# get prediction -> getPredDir(version) [w/o name]
# get prediction w/ custom name -> getPredDir(version, "name") [w/ name]
def getPredDir(version: int, name: str="") -> str:
    if (name):
        return BASE_DIR + "prediction/" + name + "_v" + str(version) + ".csv"
    return BASE_DIR + "prediction/predictions_v" + str(version) + ".csv"