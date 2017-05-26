import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt

import pickle

# Open the grid_search result

def main():
    # Load and prepare the data

    data_path = 'Bike-Sharing-Dataset/hour2.csv'
    MODEL_DIR = 'Models2/'
    rides = pd.read_csv(data_path)

    fields_to_drop = ['instant', 'dteday', 'yr', 'casual', 'registered']
    data = rides.drop(fields_to_drop, axis=1)
    test_data = data[-28*24:]
    with open('tf2_result2017-05-26', 'rb') as f:
        losses = pickle.load(f)

    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['test'], label='Test loss')
    plt.legend()


    predictions = losses['predictions']
    targets = losses['targets']

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