"""
This class is responsible for testing the churn library.

Author: Felipe Lana Machado
Date: 06/02/2022
"""
import os
import logging as log
import unittest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.churn.churn_library import Churn
from src.helpers.project_paths import DOCS_LOGS, IMAGES_EDA


class TestChurn(unittest.TestCase):

    def setUp(self) -> None:
        self.churn = Churn()
        log.basicConfig(
            filename=os.path.join(DOCS_LOGS, 'churn_library_test.log'),
            level=log.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')

    def test_import(self):
        '''
        Test data import from file.

        Raises:
            err: Raises an FileNotFoundError
        '''
        try:
            df = self.churn.import_data("bank_data.csv")
            log.info("SUCCESS on import_data: The data has been imported.")
        except FileNotFoundError as err:
            log.error("ERROR on import_data: The file wasn't found.")
            raise err

    def test_has_data(self):
        """
        Test if data import has rows and columns.

        Raises:
            err: Raises an AssertionError
        """
        df = self.churn.import_data("bank_data.csv")
        try:
            self.assertTrue(df.shape[0] > 0)
            self.assertTrue(df.shape[1] > 0)
            log.info(
                f"SUCCESS on import_data: Data has {df.shape[0]} rows and {df.shape[1]} columns.")
        except AssertionError as err:
            log.error(
                "ERROR on import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_eda(self):
        '''
        Test the perform eda function, looking to assert if there is files in the images/eda folder after the execution on the program.

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        df = self.churn.import_data("bank_data.csv")
        try:
            df = self.churn.perform_eda(df=df)
            list = os.listdir(IMAGES_EDA)
            number_files = len(list)
            self.assertTrue(number_files == 5)
            log.info(
                f"SUCCESS on perform_eda: The is {number_files} files in images EDA directory.")
        except AssertionError as err:
            log.error(
                f"ERROR on perform_eda: There is {number_files} files inside the images directory.")
            raise err

    def test_encoder_helper(self):
        '''
        Test if the columns are encoded correctly, 

        Raises:
            err: Raises an AssertionError if there is no columns that contains _churn.
        '''

        df = self.churn.import_data("bank_data.csv")
        df = self.churn.perform_eda(df=df)
        try:
            df = self.churn.encoder_helper(df=df, category_list=list(
                df.select_dtypes(include=['object']).columns), response='churn')
            has_churn_rate_columns = any(
                [True if x.find('_churn') != -1 else False for x in list(df.columns)])
            self.assertTrue(has_churn_rate_columns)
            log.info(
                f"SUCCESS on encoder_helper(): There is columns containing the name _chrun in the dataframe.")
        except AssertionError as err:
            log.error(
                f"ERROR on encoder_helper(): There isn`t any columns containing the name _chrun in the dataframe.")
            raise err

    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        df = self.churn.import_data("bank_data.csv")
        df = self.churn.perform_eda(df=df)
        df = self.churn.encoder_helper(df=df, category_list=list(
            df.select_dtypes(include=['object']).columns), response='churn')
        columns_for_training = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                'Gender_churn', 'Education_Level_churn', 'Marital_Status_churn',
                                'Income_Category_churn', 'Card_Category_churn']
        X_train, X_test, y_train, y_test = self.churn.perform_feature_engineering(
            df=df, response=columns_for_training)
        try:
            self.assertFalse(X_train.empty)
            self.assertFalse(y_train.empty)
            log.info(
                'SUCCESS on perform_feature_engineering: Train data was created and not empty.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Train data could not be created')
            raise err

        try:
            self.assertFalse(X_test.empty)
            self.assertFalse(y_test.empty)
            log.info(
                'SUCCESS on perform_feature_engineering: Test data was created and not empty.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Test data could not be created.')
            raise err

        try:
            self.assertEqual(X_train.shape[0] == y_train.shape[0])
            log.info(
                'SUCCESS on perform_feature_engineering: Train data shapes match.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Train data X and Y doesnt match.')
            raise err

        try:
            self.assertEqual(X_test.shape[0] == y_test.shape[0])
            log.info(
                'SUCCESS on perform_feature_engineering: Train data shapes match.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Test data X and Y doesnt match.')
            raise err

    def test_classification_report(self):
        """
        Tests if the function produces classification report for training and testing results and stores report as image
        in images/eda folder.

        Raises:
            err: Raises an AssertionError if there is no files in images/eda folder.
        """
        df = self.churn.import_data("bank_data.csv")
        df = self.churn.perform_eda(df=df)
        df = self.churn.encoder_helper(df=df, category_list=list(
            df.select_dtypes(include=['object']).columns), response='churn')
        columns_for_training = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                'Gender_churn', 'Education_Level_churn', 'Marital_Status_churn',
                                'Income_Category_churn', 'Card_Category_churn']
        X_train, X_test, y_train, y_test = self.churn.perform_feature_engineering(
            df=df, response=columns_for_training)
        
        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        self.churn.classification_report_image(y_train,
                                               y_test,
                                               y_train_preds_lr,
                                               y_train_preds_rf,
                                               y_test_preds_lr,
                                               y_test_preds_rf)
        list_of_files = os.listdir(IMAGES_EDA)
        images_random_forest_length = len(
            [file for file in list_of_files if 'Random_Forest' in file])

        images_logistic_regression_length = len(
            [file for file in list_of_files if 'Logistic_Regression' in file])

        try:
            self.assertEqual(images_random_forest_length, 2)
            log.info(
                f'SUCCESS on classification_report: There is two images of Random Forest (Train and Test) on images/eda folder.')
        except AssertionError as err:
            log.error(
                f'ERROR on classification_report: There isn`t two images of Random Forest (Train and Test) on images/eda folder.')
            raise err

        try:
            self.assertEqual(images_logistic_regression_length, 2)
            log.info(
                f'SUCCESS on classification_report: There is two images of Logistic Regression (Train and Test) on images/eda folder.')
        except AssertionError as err:
            log.error(
                f'ERROR on classification_report: There isn`t two images of Logistic Regression (Train and Test) on images/eda folder.')
            raise err

    def test_feature_importance_plot(self):
        """
        Tests if the function produces classification report for training and testing results and stores report as image
        in images/results folder.

        Raises:
            err: Raises an AssertionError if there is no files in images/result folder.
        """
        

        try:
            self.assertEqual(images_logistic_regression_length, 2)
            log.info(
                f'SUCCESS on classification_report: There is two images of Logistic Regression (Train and Test) on images/eda folder.')
        except AssertionError as err:
            log.error(
                f'ERROR on classification_report: There isn`t two images of Logistic Regression (Train and Test) on images/eda folder.')
            raise err

    def test_train_models(self):
        '''
        test train_models

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        df = self.churn.import_data("bank_data.csv")
        df = self.churn.perform_eda(df=df)
        df = self.churn.encoder_helper(df=df, category_list=list(
            df.select_dtypes(include=['object']).columns), response='churn')
        columns_for_training = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                'Gender_churn', 'Education_Level_churn', 'Marital_Status_churn',
                                'Income_Category_churn', 'Card_Category_churn']
        X_train, X_test, y_train, y_test = self.churn.perform_feature_engineering(
            df=df, response=columns_for_training)
        df = self.churn.train_models(
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        try:
            assert X_train.empty == False
            assert y_train.empty == False
            log.info(
                'SUCCESS on train_models: Train data was created and not empty.')

        except AssertionError as err:
            log.error(
                'ERROR on train_models: Train data could not be created')
            raise err
