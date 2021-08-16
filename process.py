# import libraries
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder, Normalizer, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit
import pickle,argparse,time

# import models
from sklearn.svm import SVR,SVC,LinearSVC
from sklearn.linear_model import LinearRegression,Lasso, Ridge, Perceptron, BayesianRidge, LogisticRegression, LassoCV, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor



class Back_To_Float(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        encoding N features and filling missing values
        too
    """
    
    def __init__(self, all_features=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],  
                        to_encode=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]):
    
        #Read in data
        self.features = all_features
        self.to_encode = to_encode

    def fit(self,X):
        #check if features are present
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset
        """
        #check if features are present
        try:
            X = X[self.features].astype('float')
        except Exception as exp:
            raise exp
        return X
    



class Encode_Feature_Label(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        encoding N features and filling missing values
        too
    """
    
    def __init__(self, all_features=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],  
                        to_encode=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]):
    
        #Read in data
        self.features = all_features
        self.to_encode = to_encode

    def fit(self,X):
        #check if features are present
        try:
            X = X[self.features].astype('string')
        except Exception as exp:
            raise exp

        self.all_encode = {each_feature : LabelEncoder().fit(X[each_feature]) for each_feature in self.to_encode}
        
        #Add 'NaN' to all classes
        # for each_feature in self.to_encode:
        #     self.all_encode[each_feature].classes_ = list(set(self.all_encode[each_feature].classes_+['NaN']))
        return self #do nothing

    def transform(self,X):
        """
            Work on the dataset
        """
        #check if features are present
        try:
            X = X[self.features].astype('string')
        except Exception as exp:
            raise exp
            
        #Replace Labels with numerical values
        for each_feature in self.to_encode:
            classes_ = self.all_encode[each_feature].classes_
            
            #remove unseen instances
            # print("class_ ", classes_)
            # X[each_feature] = X[each_feature].apply(lambda x: x if x in classes_ else 'NaN')
            X[each_feature] = self.all_encode[each_feature].transform(X[each_feature])
            
            none_index = np.where(classes_ == 'NaN')[0]
            if none_index.shape[0] >= 1:
                none_index = int(none_index)
                X[each_feature].replace(none_index,np.nan,inplace=True)
        return X
    
class Fill_Empty_Spaces_With_Values(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        Filling Missing Values with Standard Values That
        Represents Missing Values, e.g numpy.nan.
    """
    def __init__(self, all_features=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],                          
                        imputer=None
                        ):

        #Read in data
        self.features = all_features
        self.imputer = IterativeImputer(max_iter=20, random_state=0) if not imputer else imputer

    def fit(self,X):
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp
        
        self.imputer.fit(X)
        return self
        
    def transform(self,X):
        """
            Work on the dataset
        """
        
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp
            
        #Replace Missing Value With Recognized Missing Value
        return pd.DataFrame(self.imputer.transform(X), columns=self.features)
        
class Fill_Empty_Spaces_With_NaN(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data, By
        Filling Missing Values with Standard Values That
        Represents Missing Values, e.g numpy.nan.
    """
    
    def __init__(self, all_features=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],
                        find_in=["listingtitle", "conditiontitle", "sailthru_tag"],
                                        
                        find=None,
                        with_=None
                        ):
    
        #Read in data
        self.features = all_features
        self.find_in = find_in
        self.find = ['?','? ',' ?',' ? ','',' ','-',None,'None','none','Null','null',np.nan] if not find else find
        self.with_ = np.nan if not with_ else with_

    def fit(self,X):
        return self #do nothing
    def transform(self,X):
        """
            Work on the dataset
        """
        
        try:
            X = X[self.features]
        except Exception as exp:
            raise exp
            
        #Replace Missing Value With Recognized Missing Value
        X[self.find_in] = X[self.find_in].replace(self.find,self.with_)
        return X
    
    
class Round_Of_Values(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data 
        by rounding off value to nearest integer.
    """
    
    def __init__(self, all_feat=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],           
                        feat_to_round=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]):
    
        #Read in data
        self.feat_to_round = feat_to_round
        self.all_feat = all_feat

    def fit(self,X):
        return self #do nothing
    
    def transform(self,X):
        """
            Round Of Values In Features
        """
        
        try:
            X = X[self.all_feat]
        except Exception as exp:
            raise exp
            
        X[self.feat_to_round] = X[self.feat_to_round].apply(lambda x: round(x)).astype('int')
        
        return X

# class OneHotEncode_Columns(BaseEstimator, TransformerMixin):
#     """
#         This is a Class Used to Preprocess the data by
#         one hot encoding of specified features.
#     """
    
#     def __init__(self, all_feat=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],
#                  feat_to_dummy=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]):
    
#         #Read in data
#         self.feat_to_dummy = feat_to_dummy
#         self.all_feat = all_feat
#     def fit(self,X):
#         return self #do nothing
    
#     def transform(self,X):
#         """
#             One Hot Encode Some Features 
#         """
        
#         try:
#             X = X[self.all_feat]
#         except Exception as exp:
#             raise exp
            
#         X = pd.get_dummies(X,columns=self.feat_to_dummy)
#         return X

class OneHotEncode_Columns(BaseEstimator, TransformerMixin):
    """
        This is a Class Used to Preprocess the data by
        one hot encoding of specified features.
    """
    
    def __init__(self, all_feat=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],
                 feat_to_dummy=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"]):
    
        #Read in data
        self.feat_to_dummy = feat_to_dummy
        self.all_feat = all_feat
    def fit(self,X):
        try:
            X = X[self.all_feat]
        except Exception as exp:
            raise exp
        self.one_hot_encoder = OneHotEncoder().fit(X)
        return self #do nothing
    
    def transform(self,X):
        """
            One Hot Encode Some Features 
        """
        
        try:
            X = X[self.all_feat]
        except Exception as exp:
            raise exp
        X = self.one_hot_encoder.transform(X)
        X = pd.DataFrame(X.toarray(),columns=self.one_hot_encoder.get_feature_names(self.feat_to_dummy))
        return X


#############






# Data pipeline to process all incoming dataset
from sklearn.model_selection import StratifiedShuffleSplit,ShuffleSplit

class AutochekDataProcessorPipeline:
    def __init__(
                    self, path_to_body_type=None, path_to_categories=None, 
                    path_to_condition=None, path_to_listing=None, path_to_trueprices=None, 
                    features_to_extract=["make_id", "model_id", "series_id", "is_verified_dealer", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"],
                    find_missing_int=["make_id", "model_id", "series_id", "is_verified_dealer","year_of_manufacture"],
                    find_missing_str=["listingtitle", "conditiontitle", "sailthru_tag"],
                    feature_to_dummy=["make_id","model_id","series_id","is_verified_dealer","year_of_manufacture","listingtitle", "conditiontitle", "sailthru_tag"],
                    target_column=["price"]
                ):
        """Init all dataset"""
        
        self.set_paths_nd_extras(path_to_body_type, path_to_categories, path_to_condition, path_to_listing, path_to_trueprices, features_to_extract, find_missing_int, find_missing_str, feature_to_dummy, target_column)
        self.reset_attributes()
        self.load_dataset_from_diff_loc()

    def set_paths_nd_extras(self, path_to_body_type, path_to_categories, path_to_condition, path_to_listing, path_to_trueprices, features_to_extract, find_missing_int, find_missing_str, feature_to_dummy, target_column):
        """This method sets the path to all the link to get the data from"""
        self.path_to_body_type = path_to_body_type
        self.path_to_categories = path_to_categories
        self.path_to_condition = path_to_condition
        self.path_to_listing = path_to_listing
        self.path_to_trueprices = path_to_trueprices
        self.features_to_extract = features_to_extract
        self.find_missing_int = find_missing_int 
        self.find_missing_str = find_missing_str
        self.feature_to_dummy = feature_to_dummy
        self.target_column = target_column
        
        
    def get_paths_nd_extras(self):
        """This method gets the set paths from the object"""
        return (
                        self.path_to_body_type, self.path_to_categories, self.path_to_condition, 
                        self.path_to_listing, self.path_to_trueprices, self.features_to_extract, 
                        self.find_missing_int, self.find_missing_str, 
                        self.feature_to_dummy,self.target_column
                )
    
    def get_features(self):
        return self.features_to_extract
    
    def get_target(self):
        return self.target_column
    
    def get_missing_int_features(self):
        return self.find_missing_int
    
    def reset_attributes(self):
        """This method resets all the data attributes in the created object"""
        self.body_type_df = None
        self.categories_df = None
        self.condition_df = None
        self.listing_df = None
        self.trueprices_df = None
        self.dataset_output = None
        self.output_columns = None
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.currentX = None
        self.currentY = None
        
        # init pipeline flags
        self.normalizer_is_fitted = False
        self.pipeline_feature1_is_fitted = False
        self.pipeline_feature2_is_fitted = False
        self.pipeline_target_is_fitted = False
        
        #Initialize Pipeline for features
        self.process_pipeline_feature1 = Pipeline([
                        ('fill_missing_with_NaN', Fill_Empty_Spaces_With_NaN(all_features=self.get_features(),find_in=self.get_missing_int_features()+self.find_missing_str,with_='NaN')),
                        ('encode_cat_fea', Encode_Feature_Label(all_features=self.get_features(), to_encode=self.feature_to_dummy)),
                        ('int_column_to_float', Back_To_Float(all_features=self.get_features(), to_encode=self.get_missing_int_features())),
                        ('fill_missing_for_nan', Fill_Empty_Spaces_With_NaN(all_features=self.get_features(),find_in=self.get_missing_int_features()+self.find_missing_str,with_=np.nan)),
                        ('Mice_Imputer', Fill_Empty_Spaces_With_Values(all_features=self.get_features())),
                        ('Round_of_Values', Round_Of_Values(all_feat=self.get_features(),feat_to_round=self.feature_to_dummy))
                        ]) 
        
        #Initialize Pipeline for features
        self.process_pipeline_feature2 = Pipeline([
                        ('one_hot_encode', OneHotEncode_Columns(all_feat=self.get_features(), feat_to_dummy=self.feature_to_dummy))
                        ]) 
        
        #Initialize Pipeline fore target
        self.process_pipeline_target = Pipeline([
                        ('fill_missing_target', Fill_Empty_Spaces_With_NaN(all_features=self.get_target(),find_in=self.get_target(),with_=np.nan)),
                        ('Mice_Imputer_target', Fill_Empty_Spaces_With_Values(all_features=self.get_target())),
                        ]) 
        
        #Initialize Normalizer
        self.normalizer = Normalizer()
             
    def load_dataset_from_diff_loc(self):
        """This method loads all the dataset from their different path into a pandas dataframe"""
        self.body_type_df = pd.read_csv(self.path_to_body_type,sep=";") if self.path_to_body_type else None
        self.categories_df = pd.read_csv(self.path_to_categories,sep=";") if self.path_to_categories else None
        self.condition_df = pd.read_csv(self.path_to_condition,sep=";") if self.path_to_condition else None
        self.listing_df = pd.read_csv(self.path_to_listing,sep=";") if self.path_to_listing else None
        self.trueprices_df = pd.read_csv(self.path_to_trueprices,sep=";") if self.path_to_trueprices else None
        
    def _transformer(self, data, steps, normalize):
        assert self.pipeline_target_is_fitted,"Pipeline Feature1 not fitted please fit"
        target = self.process_pipeline_target.transform(data[self.get_target()])
        
        assert self.pipeline_feature1_is_fitted, "Pipeline Feature1 not fitted please fit"
        data = self.process_pipeline_feature1.transform(data[self.get_features()])
        
        if steps=="all":
            assert self.pipeline_feature2_is_fitted, "Pipeline Feature2 not fitted please fit"
            data = self.process_pipeline_feature2.transform(data[self.get_features()])
            
            if normalize:
                assert self.normalizer_is_fitted, "Normalizer not fitted please fit"
                data = self.normalizer.transform(csr_matrix(data))
                target = target.values
                
        return data, target
        
    def pipeline_transform(self, data="original",mode=None, steps="all", return_result=True, normalize=True):
        """To trsnsform data
        data: One of 4 options ('original', 'trainset', 'testset', data) 
        mode: 'train' for train mode, 'test' for test mode
        step: 'all' to follow all transform steps, 'skip2' to skip second step(one-hot encoding step)
        return_result(bool): to return processed data
        """
        if mode:
            self.set_mode(mode=mode)
        if type(data) == type("train") and data in ["original","trainset", "testset", "valset"]:
            if data=="original":
                assert type(self.dataset_output) != type(None), "original dataset is none, please run the .process_dataset method"
                self.currentX, self.currentY = self._transformer(data=self.dataset_output, steps=steps, normalize=normalize)
                if return_result:
                    return self.currentX, self.currentY
            
            elif data=="trainset":
                assert type(self.train_df) != type(None), "trainset is none, please run the .split_data method"
                self.currentX, self.currentY = self._transformer(data=self.train_df, steps=steps, normalize=normalize)
                if return_result:
                    return self.currentX, self.currentY
                            
            elif data=="testset":
                assert type(self.test_df) != type(None), "testset is none, please run the .split_data method"
                self.currentX, self.currentY = self._transformer(data=self.test_df, steps=steps, normalize=normalize)
                if return_result:
                    return self.currentX, self.currentY
                            
            elif data=="valset":
                assert type(self.val_df) != type(None), "valset is none, please run the .split_data method"
                self.currentX, self.currentY = self._transformer(data=self.val_df, steps=steps, normalize=normalize)
                if return_result:
                    return self.currentX, self.currentY
        else:
            assert type(data)==type(pd.DataFrame()), "Data has to be a type dataframe or one of the following strings 'trainset', 'testset', 'valset'."
            self.currentX, self.currentY = self._transformer(data=data, steps=steps, normalize=normalize)
            if return_result:
                return self.currentX, self.currentY

    def _fitter(self, data, steps, only_normalize):
        if only_normalize:
            # target = data[self.get_target()].values
            data, target = self._transformer(data=data, steps=steps, normalize=False)
            target = target.values
            data = self.normalizer.fit_transform(csr_matrix(data))
            self.normalizer_is_fitted = True
        else:
            target = self.process_pipeline_target.fit_transform(data[self.get_target()])
            data = self.process_pipeline_feature1.fit_transform(data[self.get_features()])
            self.pipeline_feature1_is_fitted = True
            self.pipeline_target_is_fitted = True
            
            if steps=="all":
                data = self.process_pipeline_feature2.fit_transform(data[self.get_features()])
                self.pipeline_feature2_is_fitted = True
        return data, target
    

    def pipeline_fit(self, data="original", steps="all", return_result=True, only_normalize=True):
        """To trsnsform data
        data: One of 4 options ('original', 'trainset', 'testset', data) 
        step: 'all' to follow all transform steps, 'skip2' to skip second step(one-hot encoding step)
        return_result(bool): to return processed data
        only_normalize(True|False) 
        """
        if type(data) == type("train") and data in ["original","trainset", "testset", "valset"]:
            if data=="original":
                assert type(self.dataset_output) != type(None), "original dataset is none, please run the .process_dataset method"
                self.currentX, self.currentY = self._fitter(data=self.dataset_output, steps=steps, only_normalize=only_normalize)
                if return_result:
                    return self.currentX, self.currentY
            
            elif data=="trainset":
                assert type(self.train_df) != type(None), "trainset is none, please run the .split_data method"
                self.currentX, self.currentY = self._fitter(data=self.train_df, steps=steps, only_normalize=only_normalize)
                if return_result:
                    return self.currentX, self.currentY
                            
            elif data=="testset":
                assert type(self.test_df) != type(None), "testset is none, please run the .split_data method"
                self.currentX, self.currentY = self._fitter(data=self.test_df, steps=steps, only_normalize=only_normalize)
                if return_result:
                    return self.currentX, self.currentY
                            
            elif data=="valset":
                assert type(self.val_df) != type(None), "valset is none, please run the .split_data method"
                self.currentX, self.currentY = self._fitter(data=self.val_df, steps=steps, only_normalize=only_normalize)
                if return_result:
                    return self.currentX, self.currentY
        else:
            assert type(data)==type(pd.DataFrame()), "Data has to be a type dataframe or one of the following strings 'trainset', 'testset', 'valset'."
            self.currentX, self.currentY = self._fitter(data=data, steps=steps, only_normalize=only_normalize)
            if return_result:
                return self.currentX, self.currentY

    def process_dataset(self):
        """This method joins the datasets from the different data sources into one data for training purpose"""
        # Process dataset by joining table exactly the way it was done using sql
        if type(self.output_columns) != type(None):
            self.reset_attributes()
            self.set_paths_nd_extras(*self.get_paths_nd_extras())
            self.load_dataset_from_diff_loc()
            
        assert type(self.trueprices_df) != type(None), "trueprices_df can not be None, please specify a location to get this data from."
        assert type(self.listing_df) != type(None), "listing_df can not be None, please specify a location to get this data from."
        assert type(self.condition_df) != type(None), "condition_df can not be None, please specify a location to get this data from."
        assert type(self.body_type_df) != type(None), "body_type_df can not be None, please specify a location to get this data from."
        assert type(self.categories_df) != type(None), "categories_df can not be None, please specify a location to get this data from."
        
        #left join on listing_df
        self.trueprices_df[["listingtitle"]] = self.trueprices_df.merge(self.listing_df, left_on="listing_id", right_on="id", how="left")[["title"]].copy()
        
        #left join on condition_df
        self.trueprices_df[["conditiontitle"]] = self.trueprices_df.merge(self.condition_df, left_on="condition_type_id", right_on="id", how="left")[["title"]].copy()
        
        #left join on body_type_id
        self.trueprices_df[["sailthru_tag"]] = self.trueprices_df.merge(self.body_type_df, left_on="body_type_id", right_on="id", how="left")[["sailthru_tag"]].copy()
        
        # extract columns we are interested in
        self.dataset_output = self.trueprices_df[["id", "make_id", "model_id", "series_id", "is_verified_dealer", "price", "year_of_manufacture", "listingtitle", "conditiontitle", "sailthru_tag"]].copy()
        self.output_columns = self.dataset_output.columns
        self.dataset_output = Fill_Empty_Spaces_With_NaN(all_features=self.output_columns,find_in=self.output_columns,with_="NaN").fit_transform(X=self.dataset_output)
    
    def split_data(self, split_data_train_test = True, split_data_test_val = True, train_test_ratio=0.8, test_val_ratio=0.5, return_split=False, random_seed=42):
        """This method splits dataset into train-test or train-test-val"""
        if split_data_train_test:
            (X_train,y_train), (X_test, y_test) = self.Split_Datato_Half(X=self.dataset_output, y=self.dataset_output[["price"]], train_ratio=train_test_ratio, Stratified=False, random_seed=random_seed)
            if split_data_test_val:
                (X_val,y_val), (X_test, y_test) = self.Split_Datato_Half(X=X_test, y=y_test, train_ratio=test_val_ratio, Stratified=False, random_seed=random_seed)
                self.train_df, self.test_df, self.val_df = pd.DataFrame(X_train, columns=self.output_columns), pd.DataFrame(X_test, columns=self.output_columns), pd.DataFrame(X_val, columns=self.output_columns)
                if return_split:
                    return self.train_df, self.test_df, self.val_df
            else:
                self.train_df, self.test_df = pd.DataFrame(X_train, columns=self.output_columns), pd.DataFrame(X_test, columns=self.output_columns)
                if return_split:
                    return self.train_df, self.test_df
        else:
            return None
    # def save_original_processed_data(self,original_processed_data="original_processed_data.csv"):
    #     """This method saves the result of the merge from all data sources"""
    #     assert type(self.dataset_output) != type(None), "dataset output can not be none, please run the `.process_dataset` method"
    #     self.dataset_output.to_csv(original_processed_data or "original_processed_data.csv")
        
    def save_splits(
                            self, 
                            save_ordinary_processed_data = False,
                            save_train_data = False, save_test_data = False, 
                            save_val_data = False, train_data_filename ="./data/processed/train.csv", 
                            ordinary_processed_data_filename ="./data/processed/ordinary_processed_data.csv", 
                            test_data_filename = "./data/processed/test.csv", val_data_filename = "./data/processed/val.csv",
                    ):
        """This method saves the train, test or val split"""
        if save_ordinary_processed_data:
            assert type(self.dataset_output) != type(None), "dataset output can not be none, please run the `.process_dataset` method."
            self.dataset_output.to_csv(ordinary_processed_data_filename or "./data/processed/ordinary_processed_data.csv",index=False)
            
        if save_train_data:
            assert type(self.train_df) != type(None), "train_df can not be none, please run the `.split_data` method."
            self.train_df.to_csv(train_data_filename or "./data/processed/train.csv",index=False)
            
        if save_test_data:
            assert type(self.test_df) != type(None), "test_df can not be none, please run the `.split_data` method."
            self.test_df.to_csv(test_data_filename or "./data/processed/test.csv",index=False)
            
        if save_val_data:
            assert type(self.val_df) != type(None), "val_df can not be none, please run the `.split_data` method, setting split_data_test_val=True."
            self.val_df.to_csv(val_data_filename or "./data/processed/val.csv",index=False)
            
        
    def load_splits(
                            self, 
                            load_ordinary_processed_data = False,
                            load_train_data = False, load_test_data = False, 
                            load_val_data = False, train_data_filename ="./data/processed/train.csv", 
                            ordinary_processed_data_filename ="./data/processed/ordinary_processed_data.csv", 
                            test_data_filename = "./data/processed/test.csv", val_data_filename = "./data/processed/val.csv",
                    ):
        """This method saves the train, test or val split"""
        if load_ordinary_processed_data:
            self.dataset_output = pd.read_csv(ordinary_processed_data_filename or "./data/processed/ordinary_processed_data.csv")
            
        if load_train_data:
            self.train_df = pd.read_csv(train_data_filename or "./data/processed/train.csv")
            
        if load_test_data:
            self.test_df = pd.read_csv(test_data_filename or "./data/processed/test.csv")
            
        if load_val_data:
            self.val_df = pd.read_csv(val_data_filename or "./data/processed/val.csv")
            
            
                        
    def save_pipelines_nd_normalizer(self, filename="./pipeline_nd_normalizer/pipelines_nd_normalizer"):
        dump({
                "process_pipeline_feature1":self.process_pipeline_feature1,
                "process_pipeline_feature2":self.process_pipeline_feature2,
                "process_pipeline_target": self.process_pipeline_target,
                "normalizer": self.normalizer,
                "normalizer_is_fitted":self.normalizer_is_fitted,
                "pipeline_feature1_is_fitted":self.pipeline_feature1_is_fitted,
                "pipeline_feature2_is_fitted":self.pipeline_feature2_is_fitted,
                "pipeline_target_is_fitted":self.pipeline_target_is_fitted,
        }, filename or "./pipeline_nd_normalizer/pipelines_nd_normalizer")
    
    def load_pipeline_nd_normalizer(self, filename="./pipeline_nd_normalizer/pipelines_nd_normalizer"):
        pipelines_nd_normalizer = load(filename or "./pipeline_nd_normalizer/pipelines_nd_normalizer")
        self.process_pipeline_feature1 = pipelines_nd_normalizer["process_pipeline_feature1"]
        self.process_pipeline_feature2 = pipelines_nd_normalizer["process_pipeline_feature2"]
        self.process_pipeline_target = pipelines_nd_normalizer["process_pipeline_target"]
        self.normalizer = pipelines_nd_normalizer["normalizer"]
        self.pipeline_feature1_is_fitted = pipelines_nd_normalizer["pipeline_feature1_is_fitted"]
        self.pipeline_feature2_is_fitted = pipelines_nd_normalizer["pipeline_feature2_is_fitted"]
        self.pipeline_target_is_fitted = pipelines_nd_normalizer["pipeline_target_is_fitted"]
        self.normalizer_is_fitted = pipelines_nd_normalizer["normalizer_is_fitted"]
                
    @classmethod
    def Split_Datato_Half(cls,X,y,train_ratio=0.8,Stratified=False, random_seed=42):
        """
            This Function Utilizes the Split Functions in Sklearn 
            to Split that into Two halves.
        """
        supported = [np.ndarray, pd.core.frame.DataFrame]
        if type(X) not in supported or type(y) not in supported: 
            raise ValueError(f'X is {type(X)} and y is {type(y)}, both values are expected to be either numpy array or a pandas dataframe')

        split_data = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed) if Stratified else ShuffleSplit(n_splits=1, train_size=train_ratio ,random_state=random_seed)
        
        #split the data into two halves
        try:
            X,y = X.values, y.values
        except:
            X,y = X,y

        for train_index, test_index in split_data.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return (X_train,y_train), (X_test, y_test)
        
        
        
        
def main(args=None):
    # Initialize a data pipline
    data_pipeline = AutochekDataProcessorPipeline(
                                    path_to_body_type="data/docs copy/bodytype.csv", 
                                    path_to_categories="data/docs copy/categories.csv",
                                    path_to_condition="data/docs copy/condition.csv",
                                    path_to_listing="data/docs copy/listing.csv",
                                    path_to_trueprices="data/docs copy/trueprices.csv",
                    )
        

    # process incoming data
    data_pipeline.process_dataset()

    # #split merged data into train, test, val
    data_pipeline.split_data(return_split=False)

    # # save splitted data
    data_pipeline.save_splits(save_ordinary_processed_data=True, save_train_data=True, save_test_data=True, save_val_data=True)

    # Fit Pipeline original data joined from different sources to extract categorical features.
    data_pipeline.pipeline_fit(data="original", steps='all', return_result=False, only_normalize=False)

    # Fit Normalizer on train data.
    data_pipeline.pipeline_fit(data="trainset", steps='all', return_result=False, only_normalize=True)

    # save pipelines and normalizer
    data_pipeline.save_pipelines_nd_normalizer()
    return data_pipeline

if __name__ == "__main__":
    # quick run command: reset python3 process.py 
    # args = build_argparser().parse_args()
    start = time.time()
    main(args=None)
    print(f'Process completed in {(time.time()-start)/60} mins')

    