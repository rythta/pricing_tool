import sys
import pandas as pd
import numpy as np
import joblib
# from gen_xgbr import gen_random_forest
# from gen_neural import gen_random_forest
from gen_random_forest import gen_random_forest 
import random
from datetime import timedelta

def gen_model(csv):
    data = pd.read_csv(csv)
    idx = data.groupby('title')['time_period'].idxmin()
    min_time_period = data.groupby('title')['time_period'].min().reset_index()
    data = pd.merge(data, min_time_period, on=['title', 'time_period'])
    num_data_points = data.shape[0]
    print('training random forest')
    data, model, scaler, mape, gap = gen_random_forest(data)
    title=csv.replace(".csv","_random_forest.pkl")
    joblib.dump({'model': model, 'scaler': scaler}, title)
    return data, title, mape, gap, num_data_points

if __name__ == "__main__":
    gen_model(sys.argv[1])
