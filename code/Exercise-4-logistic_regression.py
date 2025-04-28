#!/usr/bin/env python3

"""
Implement the logistic function and binary cross-entropy loss function, using titanic data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def logistic_function(x, a, b):
    return 1.0 / (1 + np.exp(-(a * x + b)))


# binary cross-entropy loss
def bcel(y, y_hat):
    y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
    return (-1 / y.size) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def read_data(path):
    df = pd.read_csv(path)
    return df


def plot(dataframe):
    col_x = dataframe.columns[0]
    col_x_val = dataframe[col_x].to_numpy(dtype=float)
    col_y = dataframe.columns[1]
    col_y_val = dataframe[col_y].to_numpy(dtype=float)

    logis_f = logistic_function(col_x_val, 0.01, -0.84)
    bce_f = bcel(col_y_val, logis_f)
    sns.scatterplot(x=col_x, y=col_y, data=dataframe)
    sns.lineplot(x=col_x, y=logis_f, color='green', data=dataframe)
    sns.lineplot(x=col_x, y=bce_f, color='red', data=dataframe)
    plt.show()


def loss(dataframe):
    col_x = dataframe.columns[0]
    col_x_val = dataframe[col_x].to_numpy(dtype=float)
    col_y = dataframe.columns[1]
    col_y_val = dataframe[col_y].to_numpy(dtype=float)

    logis_f = logistic_function(col_x_val, 0.01, -0.84)
    print(bcel(col_y_val, logis_f))


# Calling the functions
titanic_df = read_data("titanic.csv")
df_age = titanic_df.dropna(subset=['Age'])
df_age = df_age[["Age", "Survived"]]
df_class = titanic_df[["Pclass", "Survived"]]

loss(df_age)
loss(df_class)
plot(df_age)
plot(df_class)
