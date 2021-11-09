import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

if __name__ == "__main__":
    
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    
    training_df = pd.read_csv(training_file, sep=",", header=None)
    training_df.columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    'Temp3pm', 'RainToday', 'RainTomorrow']
    
    print(training_df.head(10))
    
    # testing_df = pd.read_csv(testing_file, sep=",", header=None)
    # testing_df.columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
    # 'Temp3pm', 'RainToday', 'RainTomorrow']
    
    X, y = training_df.iloc[:, :-1], training_df.iloc[:, -1]
    
    corr = 
    print()

   
    