import preprocessing
import classification
import uuid
from error_utils import InvalidUsage
from flask import Flask, request
from flask_uploads import UploadSet, configure_uploads, DATA, UploadNotAllowed
from flask_jsonpify import json, jsonify


# application
app = Flask(__name__)
app.config['UPLOADED_DATA_DEST'] = 'uploads'


# uploads
uploaded_data = UploadSet('data', DATA)
configure_uploads(app, uploaded_data)


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


"""
Step 1: Upload the CSV containing the data that will be used to train the artificial neural network. This API will
return a unique identifier for the uploaded file. This identifier should be used when referencing the uploaded data
as the CSV will not be available directly through the API.
"""
@app.route('/api/uploads/data', methods=['POST'])
def api_v0_1_upload_data():
    dataset_id = None

    if request.method == 'POST':
        if 'data' not in request.files:
            raise InvalidUsage('No data input provided in the request', status_code=500)
        data = request.files['data']
        if data is None:
            raise InvalidUsage('No data provided to store', status_code=500)
        else:
            try:
                dataset_id = str(uuid.uuid4().hex)
                uploaded_data.save(data, name=dataset_id + '.csv')
            except UploadNotAllowed:
                raise InvalidUsage('The upload was not allowed', status_code=500)

    # This method should return a unique identifier for the uploaded data, with no direct link to the data
    # file so the user from this point forward is working with our objects, not raw CSVs, etc
    return json.dumps({'success': True, 'dataset_id': dataset_id}), 200, {'ContentType': 'application/json'}


"""
Step 2: Perform data pre-processing to get the data ready for training in the artificial neural network. This API
will take everything in the CSV and store the processed data in mongo so we have it for later. This API can be called
multiple times to try different optimizations.
"""
@app.route('/api/data/preprocessing', methods=['POST'])
def api_v0_1_process_data():
    if 'dataset_id' not in request.json or request.json['dataset_id'] is None:
        raise InvalidUsage('The provided request does not contain a data set identifier. Please provide a'
                           ' \'dataset_id\' property in the request json associated with a data upload',
                           status_code=500)

    # Check to see if the features to remove has been provided
    features_to_remove = None
    if 'features_to_remove' in request.json:
        features_to_remove = request.json['features_to_remove']

    # Check to see if the features to extract has been provided
    features_to_extract = None
    if 'features_to_extract' in request.json:
        features_to_extract = request.json['features_to_extract']

    # Perform the standard pre-processing algorithm
    json = preprocessing.execute(dataset_id=request.json['dataset_id'],
                                 features_to_remove=features_to_remove,
                                 features_to_extract=features_to_extract)

    return json


"""
Step 3: Generate the neural network for the data set using the provided configuration. The options here are pretty
minimal as we don't expect an artificial network specialist to be doing this work.
"""
@app.route('/api/data/nn', methods=['POST'])
def api_v0_1_nn():
    if 'dataset_id' not in request.json or request.json['dataset_id'] is None:
        raise InvalidUsage('The provided request does not contain a data set identifier. Please provide a'
                           ' \'dataset_id\' property in the request json associated with a data upload',
                           status_code=500)

    # Check to see if the additional hidden layers have been provided
    additional_hidden_layers = 1
    if 'additional_hidden_layers' in request.json:
        additional_hidden_layers = request.json['additional_hidden_layers']

    # Check to see if the additional hidden layers have been provided
    include_dropouts = True
    if 'include_dropouts' in request.json:
        include_dropouts = request.json['include_dropouts']

    # Perform the standard pre-processing algorithm
    json = classification.build(dataset_id=request.json['dataset_id'],
                                additional_hidden_layers=additional_hidden_layers,
                                include_dropouts=include_dropouts)

    return json


"""
Step 4: Perform the training. This method returns the process identifier for the executing fit. This allows the
caller to determine the status of the training.
"""
@app.route('/api/data/training', methods=['POST'])
def api_v0_1_training():
    if 'dataset_id' not in request.json or request.json['dataset_id'] is None:
        raise InvalidUsage('The provided request does not contain a data set identifier. Please provide a'
                           ' \'dataset_id\' property in the request json associated with a data upload',
                           status_code=500)

    # Kick off the training on the moel for this data set
    json = classification.fit(dataset_id=request.json['dataset_id'],
                              batch_size=10,
                              epochs=100)

    return json


if __name__ == '__main__':
    app.run(debug=True)
