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
    test_data = data[-31*24:]

    with open('tf2_result05-27_23-28-51', 'rb') as f:
        losses_p3 = pickle.load(f)

    with open('tf2_result05-28_15-37-37', 'rb') as f:
        losses_p2 = pickle.load(f)

    with open('tf2_result05-28_23-12-51', 'rb') as f:
        losses_p1 = pickle.load(f)

    with open('tf2_result05-29_04-08-44', 'rb') as f:
        losses = pickle.load(f)

    with open('tf2_result05-29_15-42-29', 'rb') as f:
        losses_n1 = pickle.load(f)

    with open('tf2_result05-30_03-11-31', 'rb') as f:
        losses_n2 = pickle.load(f)

    with open('tf2_result05-30_19-45-26', 'rb') as f:
        losses_n3 = pickle.load(f)

    with open('tf2_result05-31_01-24-51', 'rb') as f:
        losses_n4 = pickle.load(f)

    with open('tf2_result05-31_15-45-32', 'rb') as f:
        losses_n5 = pickle.load(f)

    with open('tf2_result06-02_18-21-57', 'rb') as f:
        losses_n6 = pickle.load(f)

    with open('tf2_result06-03_04-00-26', 'rb') as f:
        losses_n7 = pickle.load(f)

    losses['train'] = losses_p3['train'] + losses_p2['train'] + losses_p1['train'] + losses['train'] + losses_n1['train'] + losses_n2['train'] + losses_n3['train'] + losses_n4['train'] + losses_n5['train'] + losses_n6['train'] + losses_n7['train']
    losses['test'] = losses_p3['test'] + losses_p2['test'] + losses_p1['test'] + losses['test'] + losses_n1['test'] + losses_n2['test'] + losses_n3['test'] + losses_n4['test'] + losses_n5['test'] + losses_n6['test'] + losses_n7['test']
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['test'], label='Test loss')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.legend()

    #
    # predictions = losses_p3['predictions']
    # targets = losses_p3['targets']
    #
    # fig, ax = plt.subplots(figsize=(64,8))
    #
    # ax.plot(predictions, label='Prediction')
    # ax.plot(targets, label='Data')
    # ax.set_xlim(right=len(predictions))
    # ax.legend()
    #
    # dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    # dates = dates.apply(lambda d: d.strftime('%b %d'))
    # ax.set_xticks(np.arange(len(dates))[12::24])
    # _ = ax.set_xticklabels(dates[12::24], rotation=45)


    predictions = losses_n7['predictions']
    targets = losses_n7['targets']
    fig, ax = plt.subplots(figsize=(64,8))

    ax.plot(predictions, label='Prediction')
    ax.plot(targets, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)

    plt.xlabel('Date')
    plt.ylabel('Bike amount')

    plt.show()

if __name__ == '__main__':
    main()