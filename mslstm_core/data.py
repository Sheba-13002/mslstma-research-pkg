# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python [conda env:refinitiv_env]
#     language: python
#     name: conda-env-refinitiv_env-py
# ---

# %%
import datetime
import math
import numpy as np
import pandas as pd
import refinitiv.data as rd
from sklearn.preprocessing import MinMaxScaler


class RefinitivDataPipeline:

    def __init__(self, ticker='INTC.O', start_date='1980-12-12', end_date='2024-02-05', sequence_length=42):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.in_scaler = MinMaxScaler()
        self.out_scaler = MinMaxScaler()

    def fetch_data(self):
        rd.open_session()
        df = rd.get_history(
            universe=[self.ticker],
            fields=['OPEN_PRC', 'HIGH_1', 'LOW_1', 'TRDPRC_1', 'ACVOL_UNS'],
            interval='1D',
            start=self.start_date,
            end=self.end_date,
            adjustments='unadjusted',
        )
        return df

    def create_dataset(self, x, y):
        x_data, y_data = [], []
        for i in range(self.sequence_length, len(x)):
            x_data.append(x[i - self.sequence_length : i, :])
            y_data.append(y[i, :])
        return np.array(x_data), np.array(y_data)

    def prepare_datasets(self, df):
        df_clean = df.copy()
        df_clean['Prediction'] = df_clean['TRDPRC_1'].shift(-1)
        df_clean = df_clean.dropna()

        x_date = df_clean.index[-math.ceil(len(df_clean) * 0.2) :]

        X_raw = df_clean.drop(columns=['Prediction'])
        y_raw = df_clean[['Prediction']]

        train_len = math.ceil(len(df_clean) * 0.6)
        val_len = math.ceil(len(df_clean) * 0.2)

        self.in_scaler.fit(X_raw.iloc[:train_len])
        self.out_scaler.fit(y_raw.iloc[:train_len])

        x = self.in_scaler.transform(X_raw)
        y = self.out_scaler.transform(y_raw)

        x_tr = x[:train_len, :]
        y_tr = y[:train_len, :]

        x_va = x[train_len - self.sequence_length : train_len + val_len, :]
        y_va = y[train_len - self.sequence_length : train_len + val_len, :]

        x_te = x[train_len + val_len - self.sequence_length :, :]
        y_te = y[train_len + val_len - self.sequence_length :, :]

        x_train, y_train = self.create_dataset(x_tr, y_tr)
        x_val, y_val = self.create_dataset(x_va, y_va)
        x_test, y_test = self.create_dataset(x_te, y_te)

        return (x_train, y_train), (x_val, y_val), (x_test, y_test), x_date
