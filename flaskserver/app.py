from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS
from sklearn import preprocessing

app = Flask(__name__)
CORS(app)

# Load the trained model once when the application starts
model2 = None
model_path2 = './rfc_model.pkl'
try:
    with open(model_path2, 'rb') as file:
        model2 = pickle.load(file)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path2}")
except EOFError:
    print(f"Error: Model file is corrupted at {model_path2}")

@app.route('/predictweather', methods=['POST'])
def predict_weather():
    data = request.get_json()
    features = [
        data['main_temp'],
        data['visibility'],
        data['wind_speed'],
        data['pressure'],
        data['humidity'],
        966,  # grnd_level
        1014  # sea_level
    ]
    prediction = model2.predict(np.array(features).reshape(1, -1))[0]
    return jsonify({'prediction': int(prediction)})


# Load the trained model
model_path = './trained models/keras/RNN_fwd.keras'
model = load_model(model_path)

# Define column names
cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
              's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

@app.route('/predictsensor', methods=['POST'])
def predict_sensor():
    # Get data from request
    data = request.get_json()
    # Create a DataFrame with the specific data point
    data_point_df = pd.DataFrame(data, columns=cols_names)
    # Normalize the 's2' feature
    data_point_df['cycle_norm'] = data_point_df['cycle']
    cols_normalize = ['s2']  # Only normalize the 's2' feature
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data_point_df[cols_normalize])
    norm_data_point_df = pd.DataFrame(min_max_scaler.transform(data_point_df[cols_normalize]), columns=cols_normalize, index=data_point_df.index)
    data_point_join_df = data_point_df[['id', 'cycle']].join(norm_data_point_df)
    data_point_df = data_point_join_df.reindex(columns=data_point_df.columns)
    # Replicate the row to create a sequence
    sequence_length = 50
    seq_cols = ['s2']  # Only use the 's2' feature
    replicated_df = pd.concat([data_point_df] * sequence_length, ignore_index=True)
    # Generate sequences
    def new_sequence_generator(feature_df, seq_length, seq_cols):
        feature_array = feature_df[seq_cols].values
        num_elements = feature_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
            yield feature_array[start:stop, :]
    new_seq_gen = list(new_sequence_generator(replicated_df, sequence_length, seq_cols))
    new_seq_set = np.array(new_seq_gen).astype(np.float32)
    # Make prediction
    new_predictions = model.predict(new_seq_set)
    # Interpret the predictions
    threshold = 0.5
    predicted_labels = (new_predictions > threshold).astype(int)
    # Output the prediction
    return jsonify({'prediction': predicted_labels.tolist()})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
