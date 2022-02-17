import os
import pandas as pd
import joblib
from src.helpers.project_paths import DOCS_MODELS
from src.churn.churn_library import Churn
from src.helpers.constants import COLUMNS_FOR_TRAINING

if __name__ == "__main__":
    churn = Churn()
    data = churn.import_data('bank_data.csv')
    data = churn.perform_eda(df=data)
    data = churn.encoder_helper(df=data, category_list=list(
        data.select_dtypes(include=['object']).columns), response='churn')
    x_train, x_test, y_train, y_test = churn.perform_feature_engineering(
        df=data, response=COLUMNS_FOR_TRAINING)
    churn.train_models(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    rfc_model = joblib.load(os.path.join(DOCS_MODELS, 'rfc_model.pkl'))
    lr_model = joblib.load(os.path.join(DOCS_MODELS, 'logistic_model.pkl'))

    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    churn.classification_report_image(y_train,
                                      y_test,
                                      y_train_preds_lr,
                                      y_train_preds_rf,
                                      y_test_preds_lr,
                                      y_test_preds_rf)

    churn.feature_importance_plot(
            model=rfc_model, x_data=pd.concat([x_train, x_test], axis=0))
    
