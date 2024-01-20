# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from datetime import datetime
from darts.metrics import mape
from darts.metrics import r2_score
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta


def create_timeseries(data, t_index):
    pd.to_datetime(data[t_index], infer_datetime_format=True)
    df.head()


def clean_dataset(data):
    print()


def split_data(data, t_index, part):
    train, val = series1[:-20], series1[-20:]
    return train


def part_indexing(data):
    p_index = data.iloc[:0, 1:]
    return p_index


def eval_model(model, train, val):
    model.fit(train)
    forecast = model.predict(len(val))
    print("model {} obtains MAPE: {:.2f}%".format(model, mape(val, forecast)))
    print("model {} obtains R2 : {:.2f}%".format(model, r2_score(val, forecast)))


def n_beats_model(series1, train):
    from darts.models import NBEATSModel
    model = NBEATSModel(input_chunk_length=24, output_chunk_length=12, random_state=42)
    model.fit(train, epochs=50, verbose=True);
    pred = model.predict(series=train, n=20)
    plt.figure(figsize=(10, 6))
    series1.plot(label="actual")
    pred.plot(label="forecast")
    pred.to_csv('prediction/prediction_nbeats_'+str(part)+'.csv')


df = pd.read_csv("data/week_data.csv")

df.head()
part_indexing(df)
create_timeseries(df, 'weeks')

for part in part_indexing(df):
    series1 = TimeSeries.from_dataframe(df, 'weeks', part)
    split_data(df, 'weeks', part)
    series1.to_csv('prediction/prediction_'+str(part)+'.csv')
    n_beats_model(series1, split_data(df, 'weeks', part))

