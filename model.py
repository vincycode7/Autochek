from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Lasso, Ridge, Perceptron, BayesianRidge, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor


class AutochekModel:
    _supported_backbones = {
            'svr':SVR(),
            'lr':LinearRegression(),
            'la':Lasso(),
            'rg':Ridge(),
            'sgd':SGDRegressor(),
            'kn':KNeighborsRegressor()
            }
    
    def __init__(self, model_back_bone=None) -> None:
        # init model back bone
        self.model_back_bone = model_back_bone
        self.current_rmse = None
        
    def compute_rmse(self,X,y, return_rmse=True):
    
        if return_rmse == True:
            return self.current_rmse
        
    def save_model(self, filename="./model/model"):
        assert type(self.model_back_bone) != type(None), "can't save empty model, please load model"
        dump(self.model_back_bone, filename)
    
    def load_model(self, filename="./model/model", return_model=False):
        self.model_back_bone = load(filename)
        if return_model:
            return self.model_back_bone
        
    def init_model_from_supported_backbones(self, backbone_type):
        assert backbone_type in self._supported_backbones.keys(), f"The specified backbone type is not in the system please choose one of the following, {self._supported_backbones.keys()}."
        self.model_back_bone = self._supported_backbones[backbone_type]
    