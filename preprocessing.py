import pandas as pd
import storage
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask_jsonpify import json


"""
Read the data in from the file path so it can be cleaned and have feature extraction applied.
"""
def read_data(file_path):
    return pd.read_csv(file_path)


"""
Get the X and y values from the data set with zero pre-processing.
"""
def get_x_and_y(dataset, columns):
    # The columns needs to be offset by 1
    columns = columns - 1

    # Assume the y value is the final column in the spreadsheet
    X = dataset.iloc[:, 0:columns].values
    y = dataset.iloc[:, columns].values

    return X, y


"""
Inpute missing values to help remove anomalies related to poor data quality rather than odd data.
"""
def clean_data(X, inpute_missing_values):
    # inpute_missing_values
    return X


"""
Remove features from the data set to remove unnecessary information.
"""
def remove_features(X, features_to_remove):
    if features_to_remove is not None:
        X = np.delete(X, features_to_remove, axis=1)

    return X


"""
Extract the ordinal features so we can appropriately process the data.
"""
def extract_ordinal_features(X, features_to_remove, features_to_extract):
    if features_to_remove is not None:
        # Sort the features in descending order
        features_to_remove.sort()

    # Only perform the feature extraction if we have features to actually extract
    if features_to_extract is not None:
        # Sort the features in descending order
        features_to_extract.sort()

        features_to_hotencode = []

        for feature_to_extract in features_to_extract:
            # Again check the features to remove is not none before checking for an offset
            if features_to_remove is not None:
                offset = 0

                # Check to see how many features were removed before this index
                for feature_to_remove in features_to_remove:
                    if feature_to_remove < feature_to_extract:
                        offset = offset + 1
                    else:
                        break

                # Apply the offset to the feature extract
                feature_to_extract = feature_to_extract - offset

            # Create the label encoder for the feature extraction
            labelencoder = LabelEncoder()
            X[:, feature_to_extract] = labelencoder.fit_transform(X[:, feature_to_extract])

            # Check to see if the result of the label encoding needs to be hot encoded
            if np.max(X[:, feature_to_extract]) > 1:
                features_to_hotencode.append(feature_to_extract)

        if len(features_to_hotencode) > 0:
            # Encode the features and put the result at the start of X
            onehotencoder = OneHotEncoder(categorical_features=features_to_hotencode)
            X = onehotencoder.fit_transform(X).toarray()

    return X


"""
Split the data into training and validation sets (~ 80/20 split) and returns X and y data sets for both.
"""
def get_training_and_validation_sets(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=0)


def get_uploaded_data(dataset_id):
    # Read the uploaded CSV data set into memory
    dataset = read_data(file_path='uploads/' + dataset_id + '.csv')

    # Get the names of the columns
    columns = list(dataset.columns.values)

    formatted_columns = []
    column_counter = 0

    # Generate the columns in the appropriate format
    for column in columns:
        key = clean_column_name(key=column)
        name = '(' + str(column_counter) + ') ' + column
        formatted_columns.append({'key': key, 'name': name})
        column_counter = column_counter + 1

    # Get the data for the columns and apply the appropriate key
    data = dataset.iloc[:, :].values
    rows = data.shape[0]
    row_counter = 0
    formatted_data = []

    # If the data has more than 100 rows, we cap it at 100
    if rows > 100:
        rows = 100

    # Go through all of the entries in the array
    while row_counter < rows:
        column_index = 0
        row = {}

        while column_index < column_counter:
            key = formatted_columns[column_index]['key']
            value = data[row_counter, column_index]

            row[key] = value

            column_index = column_index + 1

        formatted_data.append(row)

        row_counter = row_counter + 1

    # Put all of the data together for the response
    result = {'columns': formatted_columns, 'data': formatted_data}

    return json.dumps(result)


def clean_column_name(key):
    return re.sub(r'\W+', '', key).lower()


"""
Execute the data pre-processing algorithm using the provided parameters.
"""
def execute(dataset_id, features_to_remove, features_to_extract, inpute_missing_values=True):
    # Read the uploaded CSV data set into memory
    dataset = read_data(file_path='uploads/' + dataset_id + '.csv')

    # We currently assume that y is in the last column TODO Remove this assumption
    X, y = get_x_and_y(dataset=dataset, columns=dataset.shape[1])

    # Now we have the data, we want to clean it up
    # Start by removing unnecessary features that will not help the classifier
    if features_to_remove is not None:
        X = remove_features(X, features_to_remove=features_to_remove)

    # Now handle categories of data
    if features_to_extract is not None:
        X = extract_ordinal_features(X, features_to_remove=features_to_remove, features_to_extract=features_to_extract)

    # Check to see if values should be inputed
    if inpute_missing_values:
        X = clean_data(X, inpute_missing_values)

    # Split up the data into training and validation sets
    X_train, X_validate, y_train, y_validate = get_training_and_validation_sets(X, y)

    # Finally, we take the final data and scale it appropriately
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)

    # Get the scaler properties so we can persist them
    scaler = {'scale': scaler.scale_.tolist(), 'mean': scaler.mean_.tolist(), 'var': scaler.var_.tolist()}

    # Assemble the result of all of the data pre-processing
    result = {
        'X_train': X_train.tolist(),
        'y_train': y_train.tolist(),
        'X_validate': X_validate.tolist(),
        'y_validate': y_validate.tolist(),
        'scaler': scaler
    }

    # Convert the result to JSON
    result_json = json.dumps(result)

    # Store the result so we have it for future reference
    storage.save_processed(dataset_id=dataset_id, json=result_json)

    return result_json
