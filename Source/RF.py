from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from keras.utils import plot_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd

def process_data(dataset):
    columns = dataset.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
            column_contents = dataset[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            dataset[column] = list(map(convert_to_int, dataset[column]))

    return dataset


seed = 9
np.random.seed(seed)

# load datasets
input_file = r"C:\Users\Effie\Projects\Random Forest\input.csv"

dataset = pd.read_csv(input_file)

dataset.fillna(0, inplace=True)
dataset = process_data(dataset)
print("############## TRAIN SET ##############")
print(dataset.head())

X = dataset.iloc[:, :-1].astype("int32")
# Extract the target variable (last column)
Y = dataset.iloc[:, -1].astype("int32")
print("X (First 5 rows):")
print(X.head())
print("\nY (First 5 rows):")
print(Y.head())

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, dummy_y, test_size=0.3, random_state=seed)


# Random Forest
# Define a range of values for n_estimators
n_estimators_values = [2,4,6,8,10]

train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

# Train Random Forest models with different n_estimators values
""" Main parameters of RF classifier:

n_estimators: number of decision trees, high number may make more accurate and stable predictions
but may require higher training time.

min_samples_split: minimum number of samples required in a leaf node before a split is attempted,
if the number of samples is less than the required number, the node is not split.

random_state: defines the random selection of data points per decision tree and features per node.

max_features: specifies the exact number of features to consider at each split or the fraction of features to consider at each split. 

More at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

"""

for n_estimators in n_estimators_values:
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, min_samples_split=5, random_state=seed)
    rf_classifier.fit(X_train, Y_train)

    # Predictions on training set
    Y_train_pred = rf_classifier.predict(X_train)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    train_accuracies.append(train_accuracy)

    # Predictions on test set
    Y_test_pred = rf_classifier.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    test_accuracies.append(test_accuracy)

    # Loss calculation (Mean Squared Error for example)
    train_loss = np.mean((Y_train - Y_train_pred) ** 2)
    train_losses.append(train_loss)

    test_loss = np.mean((Y_test - Y_test_pred) ** 2)
    test_losses.append(test_loss)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_estimators_values, train_accuracies, label='Train')
plt.plot(n_estimators_values, test_accuracies, label='Test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Estimators')
plt.legend(loc='upper right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(n_estimators_values, train_losses, label='Train')
plt.plot(n_estimators_values, test_losses, label='Test')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Number of Estimators')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("max_features = ", rf_classifier.max_features)
print("Number of features (n_features):", X.shape[1])


