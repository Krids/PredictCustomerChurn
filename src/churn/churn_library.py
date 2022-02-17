"""
This class is to create the customer churm prediction codes.

Author: Felipe Lana Machado
Date: 04/02/2022
"""

# import libraries
import os
from PIL import ImageDraw, Image
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.helpers.project_paths import DATA_PATH, IMAGES_EDA, IMAGES_RESULTS, DOCS_MODELS
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
sns.set()


class Churn:
    '''This class performs the studies of churn.'''

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

    def perform_eda(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''
        df = df.copy()
        df['churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        df.drop(columns=['Attrition_Flag'], inplace=True)
        df['churn'] = df['churn'].astype('int8')

        plt.figure(figsize=(20, 10))
        df['churn'].hist()
        plt.savefig(os.path.join(IMAGES_EDA, 'churn_distribuition.png'))

        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.savefig(os.path.join(IMAGES_EDA, 'customer_age_distribuition.png'))

        plt.figure(figsize=(20, 10))
        df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        plt.savefig(os.path.join(
            IMAGES_EDA, 'marital_status_distribuition.png'))

        plt.figure(figsize=(20, 10))
        sns.histplot(df['Total_Trans_Ct'])
        plt.savefig(os.path.join(
            IMAGES_EDA, 'total_transaction_distribuition.png'))

        plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(os.path.join(IMAGES_EDA, 'heatmap.png'))

        return df

    def encoder_helper(self, df: pd.DataFrame, category_list: list, response: str) -> pd.DataFrame:
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name.
                [optional argument that could be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for training and testing.
        '''
        df = df.copy()
        for category in category_list:
            category_groups = df.groupby(category).mean()[response]
            df[f'{category}_{response}'] = [category_groups.loc[val]
                                            for val in df[category]]

        return df

    def perform_feature_engineering(self, df: pd.DataFrame, response: list) -> pd.DataFrame:
        '''
        input:
                df: pandas dataframe
                response: string of response name
                [optional argument that could be used for naming variables]

        output:
                x_train: x training data
                x_test: x testing data
                y_train: y training data
                y_test: y testing data
        '''
        df = df.copy()
        x_train, x_test, y_train, y_test = train_test_split(df[response],
                                                            df['churn'],
                                                            test_size=0.3,
                                                            random_state=42)
        return x_train, x_test, y_train, y_test

    def classification_report_image(self,
                                    y_train,
                                    y_test,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf) -> None:
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
        title_list = {'Random Forest Train': [y_train, y_train_preds_rf],
                      'Random Forest Test': [y_test, y_test_preds_rf],
                      'Logistic Regression Train': [y_train, y_train_preds_lr],
                      'Logistic Regression Tests': [y_test, y_test_preds_lr]}
        for title, list_of_dfs in title_list.items():
            blank_image_lrte = Image.new('RGB', (400, 150))
            draw_blankim_lrte = ImageDraw.Draw(blank_image_lrte)
            draw_blankim_lrte.text((5, 5), title)
            draw_blankim_lrte.text(
                (15, 15), classification_report(
                    list_of_dfs[0], list_of_dfs[1]))
            blank_image_lrte.save(os.path.join(
                IMAGES_EDA, f"{title.replace(' ', '_')}.png"))

    def feature_importance_plot(self, model, x_data: pd.DataFrame) -> None:
        '''
        Creates and stores the feature importances
        input:
                model: model object containing feature_importances_
                x_data: pandas dataframe of X values

        output:
                None
        '''
        importances = model.feature_importances_

        indices = np.argsort(importances)[::-1]
        names = [x_data.columns[i] for i in indices]

        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        plt.bar(range(x_data.shape[1]), importances[indices])

        plt.xticks(range(x_data.shape[1]), names, rotation=90)

        plt.savefig(os.path.join(IMAGES_RESULTS, 'Feature_Importance.png'))

    def train_models(self, x_train, x_test, y_train, y_test) -> None:
        '''
        train, store model results: images + scores, and store models
        input:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        x_train = x_train.copy()
        x_test = x_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression()

        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(x_train, y_train)

        lrc.fit(x_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

        y_train_preds_lr = lrc.predict(x_train)
        y_test_preds_lr = lrc.predict(x_test)

        self.feature_importance_plot(
            model=cv_rfc.best_estimator_, x_data=pd.concat([x_train, x_test], axis=0))

        # save best model
        joblib.dump(cv_rfc.best_estimator_, os.path.join(
            DOCS_MODELS, 'rfc_model.pkl'))
        joblib.dump(lrc, os.path.join(DOCS_MODELS, 'logistic_model.pkl'))

        self.classification_report_image(y_train,
                                         y_test,
                                         y_train_preds_lr,
                                         y_train_preds_rf,
                                         y_test_preds_lr,
                                         y_test_preds_rf)

        lrc_plot = plot_roc_curve(lrc, x_test, y_test)
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        _ = plot_roc_curve(
            cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(os.path.join(IMAGES_RESULTS, 'ROC_curve_best_models.png'))
        plt.close()

        plt.figure(figsize=(15, 8))
        explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
        shap_values = explainer.shap_values(x_test)
        shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(IMAGES_RESULTS, 'SHAP_random_forest.png'))

        # scores
        print('random forest results')
        print('test results')
        print(classification_report(y_test, y_test_preds_rf))
        print('train results')
        print(classification_report(y_train, y_train_preds_rf))

        print('logistic regression results')
        print('test results')
        print(classification_report(y_test, y_test_preds_lr))
        print('train results')
        print(classification_report(y_train, y_train_preds_lr))
