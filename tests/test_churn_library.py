"""
This class is responsible for testing the churn library.

Author: Felipe Lana Machado
Date: 06/02/2022
"""
import os
import logging as log
import unittest
import joblib
import pandas as pd

from src.helpers.constants import COLUMNS_FOR_TRAINING
from src.helpers.project_paths import DOCS_LOGS, DOCS_MODELS, IMAGES_EDA, IMAGES_RESULTS
from src.churn.churn_library import Churn


class TestChurn(unittest.TestCase):
    '''This class is responsible to perform the tests in the churn library class.'''

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
            customers_data = self.churn.import_data("bank_data.csv")
            log.info("SUCCESS on import_data: The data has been imported.")
        except FileNotFoundError as err:
            log.error("ERROR on import_data: The file wasn't found.")
            raise err

        try:
            self.assertTrue(customers_data.shape[0] > 0)
            self.assertTrue(customers_data.shape[1] > 0)
            log.info(
                (f"SUCCESS on import_data: Data has {customers_data.shape[0]} "
                "rows and {customers_data.shape[1]} columns."))
        except AssertionError as err:
            log.error(
                "ERROR on import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_perform_eda(self):
        '''
        Test the perform eda function, looking to assert if there is files in the images/eda
        folder after the execution on the program.

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        customers_data = self.churn.import_data("bank_data.csv")
        customers_data = self.churn.perform_eda(df=customers_data)
        list_of_files = os.listdir(IMAGES_EDA)
        images_of_distribuition = len(
            [file for file in list_of_files if '_distribuition' in file])
        images_of_heatmap = len(
            [file for file in list_of_files if 'heatmap' in file])
        try:
            self.assertTrue(images_of_distribuition == 4)
            log.info(
                (f"SUCCESS on perform_eda: There is {images_of_distribuition}"
                 " files for distribuition in images EDA directory."))
        except AssertionError as err:
            log.error(
                ("ERROR on perform_eda: There isn`t 4 files for distribuition"
                 " inside the images/eda directory."))
            raise err

        try:
            self.assertTrue(images_of_heatmap == 1)
            log.info(
                "SUCCESS on perform_eda: There is one file of heatmap in images EDA directory.")
        except AssertionError as err:
            log.error(
                ("ERROR on perform_eda: There isn`t one files of heatmap"
                 " inside the images/eda directory."))
            raise err

    def test_encoder_helper(self):
        '''
        Test if the columns are encoded correctly,

        Raises:
            err: Raises an AssertionError if there is no columns that contains _churn.
        '''

        customers_data = self.churn.import_data("bank_data.csv")
        customers_data = self.churn.perform_eda(df=customers_data)
        try:
            customers_data = self.churn.encoder_helper(df=customers_data, category_list=list(
                customers_data.select_dtypes(include=['object']).columns), response='churn')
            has_churn_rate_columns = any(
                [bool(x.find('_churn') != -1) for x in list(customers_data.columns)])
            self.assertTrue(has_churn_rate_columns)
            log.info(
                ("SUCCESS on encoder_helper(): There is columns containing"
                 " the name _chrun in the dataframe."))
        except AssertionError as err:
            log.error(
                ("ERROR on encoder_helper(): There isn`t any columns"
                 " containing the name _chrun in the dataframe."))
            raise err

    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        customers_data = self.churn.import_data("bank_data.csv")
        customers_data = self.churn.perform_eda(df=customers_data)
        customers_data = self.churn.encoder_helper(df=customers_data, category_list=list(
            customers_data.select_dtypes(include=['object']).columns), response='churn')
        x_train, x_test, y_train, y_test = self.churn.perform_feature_engineering(
            df=customers_data, response=COLUMNS_FOR_TRAINING)
        try:
            self.assertFalse(x_train.empty)
            self.assertFalse(y_train.empty)
            log.info(
                'SUCCESS on perform_feature_engineering: Train data was created and not empty.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Train data could not be created')
            raise err

        try:
            self.assertFalse(x_test.empty)
            self.assertFalse(y_test.empty)
            log.info(
                'SUCCESS on perform_feature_engineering: Test data was created and not empty.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Test data could not be created.')
            raise err

        try:
            self.assertTrue(x_train.shape[0] == y_train.shape[0])
            log.info(
                'SUCCESS on perform_feature_engineering: Train data shapes match.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Train data X and Y doesnt match.')
            raise err

        try:
            self.assertTrue(x_test.shape[0] == y_test.shape[0])
            log.info(
                'SUCCESS on perform_feature_engineering: Test data shapes match.')

        except AssertionError as err:
            log.error(
                'ERROR on perform_feature_engineering: Test data X and Y doesnt match.')
            raise err

    def test_train_models(self):
        '''
        This function test if the train_models method save models correctly and store the
        images of the results in the properly location.

        Raises:
            err: Raises an AssertionError if there is no files in image folders.
        '''
        customers_data = self.churn.import_data("bank_data.csv")
        customers_data = self.churn.perform_eda(df=customers_data)
        customers_data = self.churn.encoder_helper(df=customers_data, category_list=list(
            customers_data.select_dtypes(include=['object']).columns), response='churn')

        x_train, x_test, y_train, y_test = self.churn.perform_feature_engineering(
            df=customers_data, response=COLUMNS_FOR_TRAINING)

        self.churn.train_models(
            x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

        list_of_files = os.listdir(DOCS_MODELS)
        model_of_rfc = len(
            [file for file in list_of_files if 'rfc_model' in file])

        model_of_logistic = len(
            [file for file in list_of_files if 'logistic_model' in file])

        try:
            self.assertEqual(model_of_rfc, 1)
            log.info(
                ("SUCCESS on train_models: There is one model of"
                 " Random Forest on docs/models folder."))
        except AssertionError as err:
            log.error(
                ("ERROR on train_models: There isn`t one model of"
                 " Random Forest on docs/models folder."))
            raise err

        try:
            self.assertEqual(model_of_logistic, 1)
            log.info(
                ('SUCCESS on train_models: There is one model of'
                 ' Logistic Regression on docs/models folder.'))
        except AssertionError as err:
            log.error(
                ('ERROR on train_models: There isn`t one model of'
                 ' Logistic Regression on docs/models folder.'))
            raise err

        list_of_files = os.listdir(IMAGES_RESULTS)
        image_of_roc_curve = len(
            [file for file in list_of_files if 'ROC_curve_best_models' in file])

        image_of_shap_values = len(
            [file for file in list_of_files if 'SHAP_random_forest' in file])

        try:
            self.assertEqual(image_of_roc_curve, 1)
            log.info(
                ('SUCCESS on train_models: There is one image for ROC'
                 ' curve on images/results folder.'))
        except AssertionError as err:
            log.error(
                ('ERROR on train_models: There isn`t one image for'
                 ' ROC curve on images/results folder.'))
            raise err

        try:
            self.assertEqual(image_of_shap_values, 1)
            log.info(
                ('SUCCESS on train_models: There is one image for SHAP values on'
                 ' the Randon Forest on images/results folder.'))
        except AssertionError as err:
            log.error(
                ('ERROR on train_models: There isn`t one image for SHAP values on'
                 ' the Randon Forest on images/results folder.'))
            raise err

    def test_classification_report(self):
        """
        Tests if the function produces classification report for training and
        testing results and stores report as image in images/eda folder.

        Raises:
            err: Raises an AssertionError if there is no files in images/eda folder.
        """
        customers_data = self.churn.import_data("bank_data.csv")
        customers_data = self.churn.perform_eda(df=customers_data)
        customers_data = self.churn.encoder_helper(df=customers_data, category_list=list(
            customers_data.select_dtypes(include=['object']).columns), response='churn')

        x_train, x_test, y_train, y_test = self.churn.perform_feature_engineering(
            df=customers_data, response=COLUMNS_FOR_TRAINING)

        rfc_model = joblib.load(os.path.join(DOCS_MODELS, 'rfc_model.pkl'))
        lr_model = joblib.load(os.path.join(DOCS_MODELS, 'logistic_model.pkl'))

        y_train_preds_rf = rfc_model.predict(x_train)
        y_test_preds_rf = rfc_model.predict(x_test)

        y_train_preds_lr = lr_model.predict(x_train)
        y_test_preds_lr = lr_model.predict(x_test)

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
                ('SUCCESS on classification_report: There is two images of'
                 ' Random Forest (Train and Test) on images/eda folder.'))
        except AssertionError as err:
            log.error(
                ('ERROR on classification_report: There isn`t two images of'
                 ' Random Forest (Train and Test) on images/eda folder.'))
            raise err

        try:
            self.assertEqual(images_logistic_regression_length, 2)
            log.info(
                ('SUCCESS on classification_report: There is two images of'
                 ' Logistic Regression (Train and Test) on images/eda folder.'))
        except AssertionError as err:
            log.error(
                ('ERROR on classification_report: There isn`t two images of'
                 ' Logistic Regression (Train and Test) on images/eda folder.'))
            raise err

    def test_feature_importance_plot(self):
        """
        Tests if the function produces classification report for training and
         testing results and stores report as image in images/results folder.

        Raises:
            err: Raises an AssertionError if there is no files in images/result folder.
        """
        customers_data = self.churn.import_data("bank_data.csv")
        customers_data = self.churn.perform_eda(df=customers_data)
        customers_data = self.churn.encoder_helper(df=customers_data, category_list=list(
            customers_data.select_dtypes(include=['object']).columns), response='churn')

        x_train, x_test, _, _ = self.churn.perform_feature_engineering(
            df=customers_data, response=COLUMNS_FOR_TRAINING)

        rfc_model = joblib.load(os.path.join(DOCS_MODELS, 'rfc_model.pkl'))

        self.churn.feature_importance_plot(
            model=rfc_model, x_data=pd.concat([x_train, x_test], axis=0))

        list_of_files = os.listdir(IMAGES_RESULTS)
        images_random_forest_importance_length = len(
            [file for file in list_of_files if 'Feature_Importance' in file])
        try:
            self.assertEqual(images_random_forest_importance_length, 1)
            log.info(
                ('SUCCESS on feature_importance_plot: There is a image of'
                 ' Random Forest features importance on images/results folder.'))
        except AssertionError as err:
            log.error(
                ('ERROR on feature_importance_plot: There isn`t a image of'
                 ' Random Forest features importance on images/results folder.'))
            raise err
