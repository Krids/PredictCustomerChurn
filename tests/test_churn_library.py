import os
import logging as log
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.helpers.project_paths import DOCS_LOGS, IMAGES_EDA
from src.churn.churn_library import Churn

log.basicConfig(
    filename=os.path.join(DOCS_LOGS, 'churn_library_test.log'),
    level=log.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class TestChurn:

    def setup_method(self) -> None:
        self.churn = Churn()

    def test_import(self):
        '''
        Test data import from file.

        Raises:
            err: Raises an FileNotFoundError
        '''
        try:
            df = self.churn.import_data("bank_data.csv")
            log.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            log.error("INFO: Testing import_eda: The file wasn't found")
            raise err

    def test_has_data(self):
        """
        Test if data import has rows and columns.

        Raises:
            err: Raises an AssertionError
        """
        df = self.churn.import_data("bank_data.csv")
        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
            log.info(
                f"SUCCESS: Data has {df.shape[0]} rows and {df.shape[1]} columns.")
        except AssertionError as err:
            log.error(
                "ERROR: The file doesn't appear to have rows and columns")
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
            list = os.listdir(IMAGES_EDA) # dir is your directory path
            number_files = len(list)
            assert number_files == 5
            log.info(
                f"SUCCESS: The is {number_files} files in images EDA directory.")
        except AssertionError as err:
            log.error(
                f"ERROR: There is {number_files} files inside the images directory.")
            raise err

    def test_encoder_helper(encoder_helper):
        '''
        test encoder helper
        '''

    def test_perform_feature_engineering(perform_feature_engineering):
        '''
        test perform_feature_engineering
        '''

    def test_train_models(train_models):
        '''
        test train_models
        '''


if __name__ == "__main__":
    test_churn = TestChurn()
    test_churn.setup_method()
    test_churn.test_import()
    test_churn.test_has_data()
    test_churn.test_eda()