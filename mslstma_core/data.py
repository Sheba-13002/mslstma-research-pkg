# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (Refinitiv)
#     language: python
#     name: refinitiv_env
# ---

# %% [raw]
# mslstma_research_pkg

# %%
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import refinitiv.data as rd

class RefinitivDataPipeline:
    def __init__(self, ticker="INTC.O", start_date="1980-12-12", end_date="2024-02-05", sequence_length=42):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.in_scaler = MinMaxScaler()
        self.out_scaler = MinMaxScaler()

    def fetch_data(self) -> pd.DataFrame:
        rd.open_session()
        df = rd.get_history(
            universe=[self.ticker],
            fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"],
            interval="1D",
            start=self.start_date,
            end=self.end_date,
            adjustments="unadjusted"
        )
        rd.close_session()
        df['Prediction'] = df['TRDPRC_1'].shift(-1)
        return df.dropna()

    def prepare_datasets(self, df: pd.DataFrame):
        X_raw = df.drop(columns=['Prediction'])
        y_raw = df[['Prediction']]
        
        train_len = math.ceil(len(df) * 0.6)
        val_len = math.ceil(len(df) * 0.2)

        # Fit scalers strictly on training portion
        self.in_scaler.fit(X_raw.iloc[:train_len])
        self.out_scaler.fit(y_raw.iloc[:train_len])

        x = self.in_scaler.transform(X_raw)
        y = self.out_scaler.transform(y_raw)

        # Extract sequence sliding window arrays
        x_tr, y_tr = self._create_sequences(x[:train_len, :], y[:train_len, :])
        x_va, y_va = self._create_sequences(x[train_len - self.sequence_length : train_len + val_len, :], 
                                            y[train_len - self.sequence_length : train_len + val_len, :])
        x_te, y_te = self._create_sequences(x[train_len + val_len - self.sequence_length :, :], 
                                            y[train_len + val_len - self.sequence_length :, :])

        x_dates = df.index[-len(y_te):]
        return (x_tr, y_tr), (x_va, y_va), (x_te, y_te), x_dates

    def _create_sequences(self, x_data, y_data):
        X_seq, Y_seq = [], []
        for i in range(self.sequence_length, len(x_data)):
            X_seq.append(x_data[i - self.sequence_length : i, :])
            Y_seq.append(y_data[i, :])
        return np.array(X_seq), np.array(Y_seq)
