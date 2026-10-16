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
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential


def build_mslstm_model(input_shape):
    n_neurons = input_shape[0] * input_shape[1]
    model = Sequential()
    model.add(
        LSTM(n_neurons, return_sequences=True, input_shape=input_shape)
    )
    model.add(LSTM(n_neurons, return_sequences=False))
    model.add(Dense(6))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
