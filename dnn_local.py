import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import math
import time
import matplotlib.pyplot as plt

def RMSLE(predict_list, actual_list):
    return (sum((math.log(p + 1) - math.log(a + 1))**2 for p, a in zip(predict_list, actual_list))/len(predict_list))**0.5

def main():
    # Load and prepare the data
    data_path = 'Bike-Sharing-Dataset/hour2.csv'
    MODEL_DIR = 'Models2_505050/'
    rides = pd.read_csv(data_path)

    fields_to_drop = ['instant', 'dteday', 'yr', 'casual', 'registered']
    data = rides.drop(fields_to_drop, axis=1)
    # print data.head()

    continuous_features = ['temp', 'atemp', 'hum', 'windspeed', 'hr']
    categorical_features = ['season',  'holiday',  'workingday', 'weathersit', 'mnth', 'weekday']

    # Splitting the data into training, testing, and validation sets

    test_data = data[-28*24:]
    train_data = data[:-28*24]

    # Converting Data into Tensors
    LABEL_COLUMN = 'cnt'
    def input_fn(df, training = True):
        # Creates a dictionary mapping from each continuous feature column name (k) to the values of that column stored in a constant Tensor.
        continuous_cols = {k: tf.constant(df[k].values)
                           for k in continuous_features}

        # Creates a dictionary mapping from each categorical feature column name (k) to the values of that column stored in a tf.SparseTensor.
        categorical_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
            for k in categorical_features}

        # Merges the two dictionaries into one.
        feature_cols = dict(continuous_cols)
        feature_cols.update(categorical_cols)

        if training:
            # Converts the label column into a constant Tensor.
            label = tf.constant(df[LABEL_COLUMN].values)

            # Returns the feature columns and the label.
            return feature_cols, label

        # Returns the feature columns
        return feature_cols

    def train_input_fn():
        return input_fn(train_data)

    def test_input_fn():
        return input_fn(test_data, False)

    engineered_features = []

    for continuous_feature in continuous_features:
        engineered_features.append(
            tf.contrib.layers.real_valued_column(continuous_feature))

    for categorical_feature in categorical_features:
        sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
            categorical_feature, hash_bucket_size=1000)

        engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,
                                                                      combiner="sum"))
    # DNN-Regressor
    regressor = tf.contrib.learn.DNNRegressor(
        feature_columns=engineered_features, hidden_units=[200, 200, 200, 200, 200, 200, 200, 200], model_dir=MODEL_DIR, activation_fn=tf.nn.relu,
        optimizer=tf.train.AdagradOptimizer(0.075)
    )

    # Training Our Model
    step = 5000
    losses = {'train':[], 'test':[]}
    for train_time in range(1):
        start_time = time.time()

        wrap = regressor.fit(input_fn=train_input_fn, steps=step)

        end_time = time.time()

        predicted_output = regressor.predict(input_fn=lambda : input_fn(train_data, False))
        predicted_output = [i if i > 0 else 0 for i in predicted_output]
        train_loss = RMSLE(predicted_output, train_data[LABEL_COLUMN])

        predicted_output_test = regressor.predict(input_fn=test_input_fn)
        predicted_output_test = [i if i > 0 else 0 for i in predicted_output_test]
        test_loss = RMSLE(predicted_output_test, test_data[LABEL_COLUMN])

        losses['train'].append(train_loss)
        losses['test'].append(test_loss)

        print train_time, train_loss, test_loss, end_time-start_time

    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['test'], label='Test loss')
    plt.legend()

    predicted_output = regressor.predict(input_fn=test_input_fn)
    predicted_output = [i if i > 0 else 0 for i in predicted_output]
    print RMSLE(predicted_output, test_data[LABEL_COLUMN])

    predictions = predicted_output
    targets = list(test_data[LABEL_COLUMN])

    fig, ax = plt.subplots(figsize=(64,8))

    ax.plot(predictions, label='Prediction')
    ax.plot(targets, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)
    plt.show()

if __name__ == '__main__':
    main()