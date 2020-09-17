import json
from flask import Flask, request, render_template

from utilities import predict_is_american_accent, get_audio_waveform_data

# create a flask object
app = Flask(__name__, static_folder='static', template_folder='templates')

# lookup table for the audio samples and MFCC coefficient used for each country of origin example
COUNTRY_INFO_LOOKUP = {
    'United States': ('US-M-2-3.wav', [5.9578, -3.2668, 2.7247, 12.4591, -7.9111, 8.4236,
                                       -9.1457, 2.3896, 0.1328, -0.9990, 2.9071, -5.4766]),
    'United Kingdom': ('UK-F-2-11.wav', [6.2547, -1.3535, -0.5367, 9.0018, -7.6169, 12.5335,
                                         -8.1423, 3.4142, 0.1665, -4.8678, 2.8330, -5.0001]),
    'France': ('FR-F-1-9.wav', [1.8838, 2.7042, -0.9517, 6.6494, -9.4683, 8.8488,
                                -8.2215, 6.8933, 0.1302, -2.3304, 1.1166, -4.02606]),
    'Spain': ('ES-M-1-1.wav', [7.0715, -6.5129, 7.6508, 11.1508, -7.6573, 12.4840,
                               -11.7098, 3.4266, 1.4627, -2.8128, 0.8665, -5.2443]),
    'Germany': ('GE-M-1-12.wav', [2.7314, -0.5694, 2.2488, 5.4827, -6.3672, 9.24032680975559,
                                  -13.8328, 6.2909, 0.7210, -3.3219, 2.5543, -9.1622]),
    'Italy': ('IT-F-1-1.wav', [-1.8924, -3.4833, 5.1448, 7.2463, -7.1989, 8.6228,
                               -11.3789, 3.7478, 0.3794, -1.6594, 0.5963, -2.1967])
}


@app.route('/')
def entry_page():
    """
    Loads initial home page.
    """
    country_options = ['United States', 'United Kingdom', 'France', 'Spain', 'Germany', 'Italy']

    return render_template('accents.html', country_options=country_options, country='', audio_file=None)


@app.route('/country/<country>', methods=['GET', 'POST'])
def select_country(country):
    """
    Reloads page with correct audio file, map location, waveform, and MFCC coefficients for country selected.

    :param str country: Selected country.
    """
    waveform_data = get_audio_waveform_data(f'static/audio/{COUNTRY_INFO_LOOKUP[country][0]}')
    mfcc_coefficients = COUNTRY_INFO_LOOKUP[country][1]
    return render_template('accents.html',
                           country_options=[key for key in COUNTRY_INFO_LOOKUP.keys()],
                           country=country,
                           audio_file=COUNTRY_INFO_LOOKUP[country][0],
                           waveform_data_x=json.dumps(waveform_data[0]),
                           waveform_data_y=json.dumps(waveform_data[1]),
                           mfcc_coefficients=mfcc_coefficients,
                           prediction=predict_is_american_accent(mfcc_coefficients))


@app.route('/predict_accent/<country>', methods=['GET', 'POST'])
def predict_accent(country):
    """
    Reloads page after user changes MFCC coefficients with slider.

    :param str country: Selected country.
    """

    mfcc_labels = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
    mfcc = []
    # takes user input and ensures it can be turned into a floats
    for i, x in enumerate(mfcc_labels):
        user_input = request.form[x]
        float_mfcc = float(user_input)
        mfcc.append(float_mfcc)

    waveform_data = get_audio_waveform_data(f'static/audio/{COUNTRY_INFO_LOOKUP[country][0]}')
    return render_template('accents.html',
                           country_options=[key for key in COUNTRY_INFO_LOOKUP.keys()],
                           country=country,
                           audio_file=COUNTRY_INFO_LOOKUP[country][0],
                           waveform_data_x=json.dumps(waveform_data[0]),
                           waveform_data_y=json.dumps(waveform_data[1]),
                           mfcc_coefficients=mfcc,
                           prediction=predict_is_american_accent(mfcc))
