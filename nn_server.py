
import numpy as np
import pandas as pd
import sys
import math

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden1 = np.random.normal(0.0, self.hidden1_nodes**-0.5,
                                       (self.hidden1_nodes, self.input_nodes))

        self.weights_hidden1_to_hidden2 = np.random.normal(0.0, self.hidden2_nodes**-0.5,
                                       (self.hidden2_nodes, self.hidden1_nodes))

        self.weights_hidden2_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                       (self.output_nodes, self.hidden2_nodes))
        self.lr = learning_rate

        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        ### Forward pass ###
        # Hidden layer 1
        hidden1_inputs = np.dot(self.weights_input_to_hidden1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # Hidden layer 2
        hidden2_inputs = np.dot(self.weights_hidden1_to_hidden2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # Output layer
        final_inputs = np.dot(self.weights_hidden2_to_output, hidden2_outputs)
        final_outputs = final_inputs

        ### Backward pass ###
        # Output error
        output_errors = targets - final_outputs

        # Backpropagated error
        hidden2_errors = np.dot(self.weights_hidden2_to_output.T, output_errors)
        hidden2_grad = hidden2_errors * hidden2_outputs * (1 - hidden2_outputs)

        hidden1_errors = np.dot(self.weights_hidden1_to_hidden2.T, hidden2_errors)
        hidden1_grad = hidden1_errors * hidden1_outputs * (1 - hidden1_outputs)

        #  Update the weights
        self.weights_hidden2_to_output += self.lr * output_errors * hidden2_outputs.T
        self.weights_hidden1_to_hidden2 += self.lr * hidden2_grad * hidden1_outputs.T
        self.weights_input_to_hidden1 += self.lr * hidden1_grad * inputs.T


    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        ### Forward pass ###
        # Hidden layer 1
        hidden1_inputs = np.dot(self.weights_input_to_hidden1, inputs)
        hidden1_outputs = self.activation_function(hidden1_inputs)

        # Hidden layer 2
        hidden2_inputs = np.dot(self.weights_hidden1_to_hidden2, hidden1_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # Output layer
        final_inputs = np.dot(self.weights_hidden2_to_output, hidden2_outputs)
        final_outputs = final_inputs

        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)

def nonNeg(x):
    if x > 0:
        return x
    else:
        return 0

def RMSLE(predict_list, actual_list):
    return (sum((math.log(nonNeg(p) + 1) - math.log(nonNeg(a) + 1))**2 for p, a in zip(predict_list, actual_list))/len(predict_list))**0.5


def main():
    # Load and prepare the data
    data_path = 'Bike-Sharing-Dataset/hour.csv'

    rides = pd.read_csv(data_path)

    # Checking out the data
    # rides[:24*10].plot(x='dteday', y='cnt')
    # plt.show()

    # Dummy variables
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)
    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    # print data.head()

    # Scaling target variables
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std

    # Splitting the data into training, testing, and validation sets

    # Save the last 21 days
    test_data = data[-36*24:-10*24]
    data = data[:-36*24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features, targets
    # val_features, val_targets = features[-30*24:], targets[-30*24:]

    # print train_features.head()

    epochs = 5000
    learning_rate = 0.075
    hidden1_nodes = 100
    hidden2_nodes = 100
    output_nodes = 1

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)

    losses = {'train':[], 'test':[]}
    mean, std = scaled_features['cnt']
    for e in range(epochs):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        for record, target in zip(train_features.ix[batch].values,
                                  train_targets.ix[batch]['cnt']):
            network.train(record, target)

        # Printing out the training progress
        train_loss = RMSLE(network.run(train_features)[0]*std+mean, train_targets['cnt'].values*std+mean)
        # print network.run(train_features)[0]*std+mean
        test_loss = RMSLE(network.run(test_features)[0]*std+mean, test_targets['cnt'].values*std+mean)
        if not e % 500:
            sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                         + "% ... Training loss: " + str(train_loss)[:5] \
                         + " ... Test loss: " + str(test_loss)[:5] + '\n')

        losses['train'].append(train_loss)
        losses['test'].append(test_loss)

    final_test_loss_list = losses['test'][-100:]
    print 'Final test loss', sum(final_test_loss_list)/len(final_test_loss_list)

if __name__ == '__main__':
    main()