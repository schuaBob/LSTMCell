# -*- coding: utf-8 -*-
"""rnnTesting.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12mHxcdVEZjdF9M9sCv0TmxMtbe8Gqwsg
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
data = pd.read_csv('8046_2010_2019.csv')
data = data.dropna(axis=0, how='all')  # 删除表中全部為NaN的行
data = data.drop(['Unnamed: 0', 'capacity'], axis=1)  # 移除沒用的column
data['date'] = pd.to_datetime(
    data['date'], format='%Y/%m/%d')  # 轉換date成python datetime格式
data = data.set_index('date')  # 設date為index
print(data)

# 標準化
data_to_use = data['close'].values
print(data_to_use)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))
# plt.figure(figsize=(16, 7))
# plt.plot(list(range(len(data))),scaled_data)
# plt.show()

# We now define the network
# Hyperparameters used in the network

# we train 30 days a time
batch_size = 30  # how many windows of data we are passing at once
# we want to know the predicted value the day after 30 days
# how big window_size is (Or How many days do we consider to predict next point in the sequence)
window_size = 45

hidden_layer = 256  # How many units do we use in LSTM cell
clip_margin = 4  # To prevent exploding gradient, we use clipper to clip gradients below -margin or above this margin
learning_rate = 0.0006
epochs = 200

# This function is used to create the features and labels for our data set by windowing the data.


def window_data(data, window_size):
    X = []
    y = []
    i = 0
    while (i + window_size) <= len(data) - 1:
        # 用past window_size天來預測測第window_size+1天
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        i += 1
    assert len(X) == len(y)
    return X, y


X, y = window_data(scaled_data, window_size)
# 倒數第30比以前是X_train
# 倒數30天以後是y_test
X_train = np.array(X[:-30])
y_train = np.array(y[:-30])
X_test = np.array(X[-30:])
y_test = np.array(y[-30:])
print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))


print(tf.__version__)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()
# we define the placeholders
inputs = tf.compat.v1.placeholder(tf.float32, [batch_size, window_size, 1])
targets = tf.compat.v1.placeholder(tf.float32, [batch_size, 1])

#weights and implementation of LSTM cell
# LSTM weights

#Weights for the input gate
weights_input_gate = tf.Variable(
    tf.random.truncated_normal([1, hidden_layer], stddev=0.05))
weights_input_hidden = tf.Variable(tf.random.truncated_normal(
    [hidden_layer, hidden_layer], stddev=0.05))
bias_input = tf.Variable(tf.zeros([hidden_layer]))

#weights for the forgot gate
weights_forget_gate = tf.Variable(
    tf.random.truncated_normal([1, hidden_layer], stddev=0.05))
weights_forget_hidden = tf.Variable(tf.random.truncated_normal(
    [hidden_layer, hidden_layer], stddev=0.05))
bias_forget = tf.Variable(tf.zeros([hidden_layer]))

#weights for the output gate
weights_output_gate = tf.Variable(
    tf.random.truncated_normal([1, hidden_layer], stddev=0.05))
weights_output_hidden = tf.Variable(tf.random.truncated_normal(
    [hidden_layer, hidden_layer], stddev=0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

#weights for the memory cell
weights_memory_cell = tf.Variable(
    tf.random.truncated_normal([1, hidden_layer], stddev=0.05))
weights_memory_cell_hidden = tf.Variable(
    tf.random.truncated_normal([hidden_layer, hidden_layer], stddev=0.05))
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

#Output layer weigts
weights_output = tf.Variable(
    tf.random.truncated_normal([hidden_layer, 1], stddev=0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))

#function to compute the gate states


def LSTM_cell(input, state, output):

    # it=σ(Wi[ht-1,Xt]+bi)
    input_gate = tf.sigmoid(tf.matmul(
        input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)

    # ft =σ(Wf[ht-1,Xt]+bf)
    forget_gate = tf.sigmoid(tf.matmul(
        input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)

    # Ot=σ(Wo[ht-1,Xt]+bo)
    output_gate = tf.sigmoid(tf.matmul(
        input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)

    # Ct=tanh(Wi[ht-1,Xt]+bi)
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) +
                          tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)

    # St=ft *St-1+it*Ct
    state = state * forget_gate + input_gate * memory_cell

    # ht=Ot*tanh St
    output = output_gate * tf.tanh(state)

    return state, output

 #we now define loop for the network
outputs = []
for i in range(batch_size):  # Iterates through every window in the batch

    #for each batch I am creating batch_state as all zeros and output for that window which is all zeros at the beginning as well.
    batch_state = np.zeros([1, hidden_layer], dtype=np.float32)
    batch_output = np.zeros([1, hidden_layer], dtype=np.float32)

    #for each point in the window we are feeding that into LSTM to get next output
    for ii in range(window_size):
        batch_state, batch_output = LSTM_cell(tf.reshape(
            inputs[i][ii], (-1, 1)), batch_state, batch_output)

    #last output is conisdered and used to get a prediction
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
print(outputs)

#we define the loss
losses = []

for i in range(len(outputs)):
    losses.append(tf.compat.v1.losses.mean_squared_error(
        tf.reshape(targets[i], (-1, 1)), outputs[i]))

loss = tf.reduce_mean(losses)

#we define optimizer with gradient clipping
gradients = tf.gradients(loss, tf.compat.v1.trainable_variables())
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
trained_optimizer = optimizer.apply_gradients(
    zip(gradients, tf.compat.v1.trainable_variables()))

#we now train the network
session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())
for i in range(epochs):
    trained_scores = []
    ii = 0
    epoch_loss = []
    while(ii + batch_size) <= len(X_train):
        X_batch = X_train[ii:ii+batch_size]
        y_batch = y_train[ii:ii+batch_size]

        o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={
                              inputs: X_batch, targets: y_batch})

        epoch_loss.append(c)
        trained_scores.append(o)
        ii += batch_size
    print('Epoch {}/{}'.format(i, epochs),
          ' Current loss: {}'.format(np.mean(epoch_loss)))

# training data
sup = []
for i in range(len(trained_scores)):
    for j in range(len(trained_scores[i])):
        sup.append(trained_scores[i][j][0])

tests = []
i = 0
while i+batch_size <= len(X_test):

    o = session.run([outputs], feed_dict={inputs: X_test[i:i+batch_size]})
    i += batch_size
    tests.append(o)

tests_new = []
for i in range(len(tests)):
    for j in range(len(tests[i][0])):
        tests_new.append(tests[i][0][j])

test_results = []
for i in range(2340):
    if i >= 2310:
        test_results.append(tests_new[i-2310])
    else:
        test_results.append(None)

test_predict = test_results[-30:]
# 老師規定的acc
# 標準化後的
acc = np.sum(np.abs(list(y_test[i] - test_predict[i]
                         for i in range(len(y_test)))))/len(y_test)
# 標準化前的
accInverse = np.sum(np.abs(list(scaler.inverse_transform(y_test)[
                    i] - scaler.inverse_transform(test_predict)[i] for i in range(len(y_test)))))/len(y_test)
print(acc)
print(accInverse)


# 印出最後30天的結果

y_predict = np.reshape(test_predict, (-1, 1))
plt.figure(figsize=(16, 7))
print(len(y_test))
print(len(y_predict))
plt.plot(scaler.inverse_transform(y_test), label='Original data')
# plt.plot(day[30:],sup, label='Training data')
plt.plot(scaler.inverse_transform(y_predict), label='Predict data')
plt.grid(linestyle=":", color="r")
plt.legend()
plt.show()
