from keras.layers import LSTM, Flatten, Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import statsmodels as sm
import pandas as pd
import numpy as np


def fit_mlp_nnet(security_name, forward=21):

    # --- PRE-PROCESS AND LOAD THE DATA ---

    # Read the raw data from the CSV
    data = pd.read_csv("indices.csv")
    # Set the dates column as the index
    data = data.set_index(data.columns[0])

    # Subset the data from 1950 - present
    data = data["1950-01-11":]

    # Indices included
    # - INDU Index: DJIA Industrials Index
    # - SPX Index: S&P 500 Index
    # - AS30 Index: Australian top 30
    # - SPTSX Index: Candadian S&P Index
    # - TPX Index: Japanese Index??
    # - TRAN Index: DJIA Transport Index

    # Subset by the specified column
    data = data[security_name]

    # # Plot the data
    # data.plot()
    # plt.show()

    # Compute the percentage change
    diffs = data.pct_change()

    # Function to convert to log return
    def to_log(x):
        return np.log(x + 1.)

    # Compute the logarithmic returns
    diffs = diffs.apply(to_log)

    # Create the rolling sum through the data
    cum_diffs = diffs.rolling(window=forward).sum()

    # --- SPLIT THE DATA INTO INPUTS AND OUTPUTS ---

    # Work out how many days we have
    n = len(diffs.index)

    # TODO: clean the inputs
    # - Try scaling the data between 0 and 1 (so that the derivatives aren't too small)
    # - Try standardizing the data (subtract mean and divide by stdev)
    # - Try smoothing the data (optional)
    # - Consider trying out classification rather than regression

    # Split the data into inputs and outputs
    inputs_ts = diffs[0:n-forward].values
    outputs_ts = cum_diffs[forward:n].values

    # Specify the window size
    window = 252

    # Store the inputs & outputs
    inputs, outputs = [], []

    for i in range(252, len(outputs_ts)):
        # Batch the data into 252 day windows
        # - Inputs: the previous 252 days
        # - Outputs: the return 5 days in the future
        inputs.append(inputs_ts[i-window:i])
        outputs.append(outputs_ts[i])

    # What we want to train the model on
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    # --- SPLIT THE INPUTS AND OUTPUTS ---

    # TODO: Fix train & test
    # Patterns included in the train and test subsets
    # should be picked randomly rather than chronologically

    # Work out size of training and set subsets
    train_size = int(0.7 * len(inputs))
    len_inputs = int(len(inputs))

    # Get the inputs for train and test
    inputs_train = inputs[0:train_size]
    inputs_test = inputs[train_size:len_inputs-2]
    inputs_oos = inputs[len_inputs-1]

    # Get the outputs for train and test
    outputs_train = outputs[0:train_size]
    outputs_test = outputs[train_size:len_inputs-2]
    outputs_oos = outputs[len_inputs-1]

    # --- BUILD THE NEURAL NETWORK MODEL ---

    # TODO: try different models
    # 1. Simple Recurrent Neural Network
    # 2. Long Short Term Memory Neural Network (LSTM)
    # 3. Gated Recurrent Neural Network (GRU)
    # NOTE: you will struggle with shapes. Just keep at it!

    # Specify the input layer for the neural network
    input_layer = Input(shape=(window,))

    layers = [input_layer]

    # TODO: try different architectures!!!
    layers.append(Dense(units=50, activation='relu')(layers[-1]))
    layers.append(Dense(units=50, activation='relu')(layers[-1]))
    layers.append(Dense(units=50, activation='relu')(layers[-1]))

    # Specify the final output layer for the neural network
    output_layer = Dense(units=1, activation='tanh')(layers[-1])

    # Build the neural network model by specifying inputs and outputs
    nnet = Model(inputs=input_layer, outputs=output_layer)

    # Compile the neural network to a TF computational graph
    nnet.compile(optimizer="adam", loss="mse")
    nnet.summary()

    # --- FIT THE NEURAL NETWORK ---

    # TODO: Add callbacks
    # - Add an early stopping callback

    training_process = nnet.fit(
        x=inputs_train,
        y=outputs_train,
        shuffle=True,
        epochs=50,
        batch_size=64,
        verbose=True,
        callbacks=[],
        validation_data=[
            inputs_test,
            outputs_test
        ]
    )

    # Get the prediction from the neural network
    oos_prediction = nnet.predict(np.array([inputs_oos]))
    oos_prediction = float(oos_prediction.flatten())

    # Print out the prediction vs. actual
    print("-" * 120)
    print("PREDICTED", oos_prediction)
    print("ACTUAL", outputs_oos)
    print("-" * 120)

    # TODO: print statistics
    # - Print accuracy & errors on test set (scikit.metrics)


fit_mlp_nnet("INDU Index")
