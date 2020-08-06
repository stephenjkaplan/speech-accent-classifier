# Speech Accent Classifier

Play around with the model and make predictions using the [web app](https://accent-identification.appspot.com/)
I made using Flask.

#### Description
Classifying audio of human speech into various accents/countries of origin using 
[MFCC coefficients](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) extracted from audio .wav files.

The popularity of a track is algorithmically calculated, and is a combination of how many plays a track has and 
how recent those plays are. This was developed over a 2-week span in August 2020 as a project for 
the [Metis](https://thisismetis.com) data science program.

The model in `flask_app/static/sklearn_models/final_model.pkl` is an ensemble of K-Nearest Neighbor and Logistic 
Regression models. The overall predictive accuracy of the model is `0.89` and it has an ROC AUC score of `0.95`. The 
blog post about it is  [here](https://stephenjkaplan.github.io/).

#### Data Source
Fokoue, E. (2020). [UCI Machine Learning Repository - Speaker Accent Recognition Data Set](https://archive.ics.uci.edu/ml/datasets/Speaker+Accent+Recognition). Irvine, CA: University of California, School of Information and Computer Science.


#### File Contents
* `flask_app/` contains all of the contents for the web application. Running `flask_app/views.py` will launch the 
   Flask app.
* `notebooks/` contains the Jupyter Notebook used to do all data analysis and modeling, as well as an accompanying 
   file of Python functions.

#### Dependencies

The contents of `/notebooks` can't be fully run because the Jupyter Notebook connects to a remote SQL database. In order 
to install the dependencies for the flask app, run:

`pip install -r requirements.txt`

in `/flask_app`.