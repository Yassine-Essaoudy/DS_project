import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_obj

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self, data_path, target=None):
        try:
            df = pd.read_csv(data_path)
            if target:
                df = df.drop(target, axis=1)
            
            # Identify numerical columns
            logging.info('done reading file')
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

            # Identify categorical columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logging.info(f'Numerical Columns:{numerical_columns}/n Categorical Columns: {categorical_columns}')

            num_pipline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy="mean")),
                    ('scaler', StandardScaler())]
            )
            
            cat_pipline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipline', num_pipline, numerical_columns),
                ('cat_pipline', cat_pipline, categorical_columns)
            ])
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_transformation(self, train_path, test_path, target):
        
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            logging.info('loaded training and testing datasets')
            
            data_path = os.path.join('artifacts', 'data.csv')
            preprocessor_obj = self.get_data_transformer_object(data_path=data_path, target=target)
            
            logging.info('preprocessor object obtained')
            
            train_input_features = train_data.drop(target, axis=1)
            train_target = train_data[target]
            
            test_input_features = test_data.drop(target, axis=1)
            test_target = test_data[target]
            
            logging.info('initiating the preprocessor')
            
            train_input_features_arr = preprocessor_obj.fit_transform(train_input_features)
            test_input_features_arr = preprocessor_obj.transform(test_input_features)
            
            train_arr = np.c_[train_input_features_arr, np.array(train_target)]
            test_arr = np.c_[test_input_features_arr, np.array(test_target)]
            
            save_obj(
                self.data_transformation_config.preprocessor_obj_path,
                preprocessor_obj
            )
            
            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__  == '__main__':
    model = DataTransformation()
    model.get_data_transformer_object()
