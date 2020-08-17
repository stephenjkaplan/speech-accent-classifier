import os
import joblib
import numpy as np
import librosa

# read in the model
model_path = 'static/sklearn_models/final_model.pkl'
if os.getcwd() != '/Users/stephendev/PycharmProjects/speech-accent-classifier/flask_app':
    model_path = '/app/' + model_path

MODEL = joblib.load(model_path)


def predict_is_american_accent(features):
    """
    Uses trained model to predict if .wav file represented by the features contains a speaker with an American accent.

    :param list features: List of floats representing the 12 MFCC Features (x1, x2, ..., x12).
    :return: "American" or "Not American"
    :rtype: str
    """

    # engineer features
    features_engineered = features[:8] + features[8+1:]

    # make a prediction
    prediction = MODEL.predict(np.array(features_engineered).reshape(1, -1))

    return 'American' if prediction[0] else 'Not American'


def get_audio_waveform_data(audio_file):
    """
    Loads .wav file and extracts audio time series.

    :param str audio_file: Filename for .wav file to load.
    :return: Tuple containing audio time series.
    :rtype:tuple
    """
    y, sr = librosa.load(audio_file)

    return list(range(len(y))), list(y.astype(float))
