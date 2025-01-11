import nbimporter
import DA_Assignment
from striprtf.striprtf import rtf_to_text
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')
import json
import pandas as pd
import numpy as np
def pipeline(rtf_file_path, output_json_path, csv_path):
    rtf_file = DA_Assignment.convert_rtf_json(rtf_file_path, output_json_path)
    with open(output_json_path, 'r') as file:
        data = json.load(file)
    df = DA_Assignment.load_data(csv_path)
    target_predictive_type = DA_Assignment.extracting_target_predictive_type(data)
    y = df[target_predictive_type['target']]
    X = df.drop(target_predictive_type['target'], axis=1)
    encode_feature = DA_Assignment.encode_categorical_features(X)
    mis = DA_Assignment.handle_missing_values(df, data)
    reduction = DA_Assignment.feature_reduction(X, y, data, target_predictive_type, encode_feature)
    selected_models = DA_Assignment.select_models(data, target_predictive_type)
    print("\nFinal Selected Models with Parameters:")
    print("=" * 50)
    for model_name, model_config in selected_models.items():
        print(f"\n{model_name}:")
        for param_name, param_value in model_config.items():
            if param_name != "model_name":
                print(f"- {param_name}: {param_value}")
    parameter = DA_Assignment.parse_hyper(data)
    predict = DA_Assignment.fit_and_predict_with_tuning(reduction, y, selected_models, parameter)

    return predict


pipe = pipeline("algoparams_from_ui.json.rtf", "algoparams_from_ui.json", "iris.csv")
