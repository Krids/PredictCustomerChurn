"""
This class is responsible for testing the churn library.

Author: Felipe Lana Machado
Date: 06/02/2022
"""
import os
import logging as log
import unittest
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
        Test perform eda function

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        df = self.churn.import_data("bank_data.csv")
        try:
            self.churn.perform_eda(df=df)
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
        df['churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        df.drop(columns=['Attrition_Flag'], inplace=True)
        df['churn'] = df['churn'].astype('int8')
        try:
            df = self.churn.encoder_helper(df=df, category_list=list(df.select_dtypes(include=['object']).columns), response='churn')
            has_churn_rate_columns = any([True if x.find('_churn') != -1 else False for x in list(df.columns)])
            self.assertTrue(has_churn_rate_columns)
            log.info(
                f"SUCCESS on encoder_helper(): There is columns containing the name chrun_rate in the dataframe.")
        except AssertionError as err:
            log.error(
                f"ERROR on encoder_helper(): There isn`t any columns containing the name chrun_rate in the dataframe.")
            raise err

    def test_perform_feature_engineering(perform_feature_engineering):
        '''
        test perform_feature_engineering

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''

    def test_train_models(train_models):
        '''
        test train_models

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
