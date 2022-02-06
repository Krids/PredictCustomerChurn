"""
This class is to create the customer churm prediction codes.

Author: Felipe Lana Machado
Date: 04/02/2022
"""

# import libraries
from src.helpers.project_paths import DATA_PATH, DOCS_LOGS, IMAGES_EDA, IMAGES_RESULTS, DOCS_MODELS
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import shap
import joblib
import logging as log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Churn:

    def __init__(self) -> None:
        pass

    def import_data(self, file_name: str) -> pd.DataFrame:
        '''
        returns dataframe for the csv found in data folder with the name of file_name.

        input:
                file_name: the name of the csv file.
        output:
                df: pandas dataframe
        '''
        df = pd.read_csv(os.path.join(DATA_PATH, file_name))
        return df

    def perform_eda(self, df: pd.DataFrame) -> None:
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

        plt.figure(figsize=(20,10)) 
        df['Churn'].hist()
        plt.savefig(os.path.join(IMAGES_EDA, 'churn_distribuition.png'))

        plt.figure(figsize=(20,10)) 
        df['Customer_Age'].hist()
        plt.savefig(os.path.join(IMAGES_EDA, 'customer_age_distribuition.png'))

        plt.figure(figsize=(20,10)) 
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.savefig(os.path.join(IMAGES_EDA, 'marital_status_distribuition.png'))

        plt.figure(figsize=(20,10)) 
        sns.histplot(df['Total_Trans_Ct'])
        plt.savefig(os.path.join(IMAGES_EDA, 'total_transaction_distribuition.png'))

        plt.figure(figsize=(20,10)) 
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        plt.savefig(os.path.join(IMAGES_EDA, 'heatmap.png'))


    def encoder_helper(self, df, category_lst, response):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        pass

    def perform_feature_engineering(self, df, response):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could be used for naming variables or index y column]

        output:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

    def classification_report_image(self,
                                    y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        pass

    def feature_importance_plot(self, model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        pass

    def train_models(self, X_train, X_test, y_train, y_test):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        pass
