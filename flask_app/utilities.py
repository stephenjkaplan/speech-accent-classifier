import joblib
import numpy as np
import librosa

# read in the model
MODEL = joblib.load("static/sklearn_models/final_model.pkl")


def predict_is_american_accent(features):

    # engineer features
    features_engineered = features[:8] + features[8+1:]

    # make a prediction
    prediction = MODEL.predict(np.array(features_engineered).reshape(1, -1))
    probabilities = MODEL.predict_proba(np.array(features_engineered).reshape(1, -1))

    return ('American' if prediction[0] else 'Not American',
            probabilities[0][1] if prediction[0] else probabilities[0][0])


def get_audio_waveform_data(audio_file):
    """

    :param audio_file:
    :return:
    """

    y, sr = librosa.load(audio_file)

    return list(range(len(y))), list(y.astype(float))
