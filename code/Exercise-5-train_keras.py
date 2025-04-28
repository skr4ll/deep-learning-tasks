# Exercise 5 skr4ll
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

# data
titanic_df = pd.read_csv("titanic.csv").dropna(subset=['Age'])
x = titanic_df[["Age", "Pclass"]].to_numpy(dtype=float)
y = titanic_df["Survived"].to_numpy()

# model
model = Sequential()
model.add(Input(shape=(2,)))
model.add(Dense(2, use_bias=True, activation="relu"))
model.add(Dense(2, use_bias=True, activation='relu'))
model.add(Dense(1, use_bias=True, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="sgd")
model.summary()

# weights before train
print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())

# train
model.fit(
    x=x,
    y=y,
    batch_size=9,
    epochs=35,
    verbose=1,
    validation_split=0.3,
    shuffle=True
)

# weights after train
print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())

# the models val_loss converges to around 0.668 after around 35 epochs with the above values, with the 4th
# decimal digit jumping around a bit afterwards
