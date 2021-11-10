import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_prior(x, y):
    classes = sorted(list(x[y].unique()))
    prior = []
    for i in classes:
        prior.append(len(x[x[y]==i])/len(x))
    return prior

def calculate_likelihood_gaussian(X, feature, value, Y, label):
    X = X[X[Y]==label]
    mean = X[feature].mean()
    std = X[feature].std()
    p_x_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((value-mean)**2 / (2 * std**2 )))
    return p_x_y

def calculate_likelihood_categorical(X, feature, value, Y, label):
    X = X[X[Y]==label]
    p_x_y = len(X[X[feature]==value]) / len(X)
    return p_x_y

def naive_bayes_gaussian(df, X, Y):
    # get feature names
    features = list(df.columns)[:-1]

    # calculate prior
    prior = calculate_prior(df, Y)

    Y_pred = []
    # loop over every data sample
    index = 0
    for x in X:
        print(index)
        index += 1
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                if i==0 or i==6 or i==8 or i==9 or i == 20:
                    likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])
                else:
                    likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])
                        

        # calculate posterior probability (numerator only)
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 

def accuracy(prediction, target):
    return np.sum(prediction == target)/len(target)

if __name__ == "__main__":
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    test = pd.read_csv(test_file, sep=",", header=None)
    test.columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'RainTomorrow']

    train = pd.read_csv(train_file, sep=",", header=None)
    train.columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'RainTomorrow']
    train = train.sample(frac=0.1)

    X_test = test.iloc[:,:-1].values
    Y_test = test.iloc[:,-1].values
    Y_pred = naive_bayes_gaussian(train, X=X_test, Y="RainTomorrow")

    Y_test_formatted = [1]*len(Y_pred)
    for i in range(len(Y_pred)):
        if Y_test[i] == ' Yes':
            Y_test_formatted[i] = 1
        else:
            Y_test_formatted[i] = 0
    np.array(Y_test_formatted)

    print(Y_pred)
    print(Y_test_formatted)
    print(accuracy(Y_pred, Y_test_formatted))


