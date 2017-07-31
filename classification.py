import storage
import h5py
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from flask_jsonpify import json
from numpy import array
from rq import Queue


def load_classifier(dataset_id):
    classifier_json = storage.get_nn(dataset_id=dataset_id)

    # Load the pre-built classifier and kick off the training
    classifier = model_from_json(classifier_json)

    return classifier


def load_data(dataset_id):
    # Load the data for the fit
    data_json = storage.get_processed(dataset_id=dataset_id)
    data = json.loads(data_json)

    # Convert the arrays to numpy arrays
    X_train = array(data.get('X_train'))
    y_train = array(data.get('y_train'))
    X_validate = array(data.get('X_validate'))
    y_validate = array(data.get('y_validate'))

    # Get the scaler so we have access to the scaling used on this dataset
    scaler = data.get('scaler')

    return X_train, y_train, X_validate, y_validate, scaler


def build(dataset_id, additional_hidden_layers=1, include_dropouts=True):
    # Load the data we'll use for the neural network
    X_train, y_train, X_validate, y_validate, scaler = load_data(dataset_id=dataset_id)

    # Get the dimensions of the training data so we can correctly set the input nodes
    input_dimensions = X_train.shape[1]
    # We only support one output dimension at the moment
    output_dimensions = 1

    # We use the standard approach for unit determination
    units = (input_dimensions + output_dimensions) / 2

    classifier = Sequential()
    classifier.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim =input_dimensions))

    if include_dropouts:
        classifier.add(Dropout(rate=0.1))

    if additional_hidden_layers >= 1:
        counter = 0

        # Create additional layers as requested
        while counter < additional_hidden_layers:
            counter = counter + 1
            classifier.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
            if include_dropouts:
                classifier.add(Dropout(rate=0.1))

    classifier.add(Dense(units=output_dimensions, kernel_initializer='uniform', activation='sigmoid'))

    # Get the JSON for the configured classifier
    classifier_json = classifier.to_json()

    # Before returning the classifier, we need to save it
    storage.save_nn(dataset_id=dataset_id, json=classifier_json)

    return classifier_json


def fit(dataset_id, batch_size=10, epochs=100):
    # Load the data we'll use for the training
    X_train, y_train, X_validate, y_validate, scaler = load_data(dataset_id=dataset_id)

    # Start the training in a queued thread
    result = queue_work(function_to_queue=fit_async,
                        args=(dataset_id, X_train, y_train, batch_size, epochs),
                        dataset_id=dataset_id)

    return json.dumps(result)


def fit_async(dataset_id, X_train, y_train, batch_size, epochs):
    # Load the classifier and start training
    classifier = load_classifier(dataset_id=dataset_id)

    # Compile with the standard settings as we'll leave setting variation to the optimization
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Kick off the training
    classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Save the trained classifier back to the file system
    classifier.save('uploads/' + dataset_id + '_nn_trained.h5')

    return True


def queue_work(function_to_queue, args, dataset_id):
    # Set a default response
    response = {'dataset_id': dataset_id, 'job_status': None, 'logs': None}

    # Tell RQ what Redis connection to use
    queue = Queue(connection=storage.get_database())

    # Check to see if this data set is already running
    job = queue.fetch_job(dataset_id)

    if job is not None:
        if job.result is None:
            response['job_status'] = 'The training for the provided data set is currently running'

            # Get the logs from Redis
            logs_json = storage.get_training_logs(dataset_id=dataset_id)
            if logs_json is not None:
                response['logs'] = json.loads(logs_json)
        elif job.result is True:
            response['job_status'] = 'The training has been completed'
        else:
            response['job_status'] = 'Something has gone wrong with the training'
    else:
        # Gets a list of job IDs from the queue
        if queue.job_ids is not None and len(queue.job_ids) > 10:
            response['job_status'] = 'The server currently has 10 jobs running. Please try again later'
        else:
            # Queue the training function as defined in the classification and set an 1 hour timeout
            # As the result timeout is set to 1 hour, this means training for a data set can only be performed once
            # per hour as the cache will not have cleared the result
            queue.enqueue_call(func=function_to_queue, args=args, timeout='1h', job_id=dataset_id, result_ttl=3600)

            # Set the response to tell the caller that the training has started
            response['job_status'] = 'The training has been started successfully'

    return response


def evaluate(X, y, classifier_build_fn, batch_size=10, epochs=100):
    classifier = KerasClassifier(build_fn=classifier_build_fn, batch_size=batch_size, epochs=epochs)
    accuracies = cross_val_score(estimator=classifier, X=X, y=y, cv=10, n_jobs=-1)

    return {'mean': accuracies.mean(), 'variance':accuracies.std()}


def optimize(dataset_id, X, y, classifier_build_fn, batch_size_options=[25, 32], epochs_options=[100, 500]):
    classifier = KerasClassifier(build_fn=classifier_build_fn)

    parameters = {'batch_size': batch_size_options,
                  'epochs': epochs_options,
                  'optimizer': ['adam', 'rmsprop'],
                  'dataset_id': [dataset_id]}

    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10)

    grid_search = grid_search.fit(X, y)

    return {'best_parameters': grid_search.best_params_, 'best_accuracy': grid_search.best_score_}
