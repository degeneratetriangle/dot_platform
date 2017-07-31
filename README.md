# DOT Platform
A platform for building various neural networks without writing any code.

## Pre-requisites

* Redis

## Running

In a terminal window to start the server:
```bash
python server.py
```

In another terminal window to start the RQ worker:
```bash
rq worker default
```

## Trying it out

### Step 1: Upload the data set

1. Open index.html (directly from the file in /app)
1. Upload the test CSV file in /uploads (this will be uploaded to the same location, but with a new dataset_id)
1. Note the dataset_id in the response

Now we can run through the sequence of steps to preprocess the data, create the neural network, and train it.

### Step 2: Preprocess the data set

```bash
POST: http://127.0.0.1:5000/api/data/preprocessing
```
```json
{
  "dataset_id": "{dataset_id}",
  "features_to_remove": [0, 1, 2],
  "features_to_extract": [4, 5]
}
```

### Step 3: Create the neural network

```bash
POST: http://127.0.0.1:5000/api/data/nn
```
```json
{
  "dataset_id": "{dataset_id}",
  "additional_hidden_layers": 1,
  "include_dropouts": true
}
```

### Step 4: Train the neural network

```bash
POST: http://127.0.0.1:5000/api/data/training
```
```json
{
  "dataset_id": "{dataset_id}",
}
```

This endpoint can be hit multiple times. The platform only allows a data set/neural network pair to be trained once per hour. It will also timeout if the training takes more than 1 hour to complete.
