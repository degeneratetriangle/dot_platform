import preprocessing
import classification


# Perform the standard pre-processing algorithm
result = preprocessing.execute('abcd',
                               features_to_remove=[0, 1, 2],
                               features_to_extract=[4, 5])

classifier = classification.build(dataset_id='abcd',
                                  input_dimensions=11,
                                  output_dimensions=1,
                                  additional_hidden_layers=0)

classification.fit(X=result.X_train, y=result.y_train, classifier=classifier)
