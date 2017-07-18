"""
Two models for hit prediction with logistic regression and random forest classifier.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.linear_model import LogisticRegression

def get_xy(df):
    y = df.Hit.values.tolist()
    fields = [u'duration_ms_track', 'album_year', 'name_track_len',
           u'acousticness_tf', u'danceability_tf', u'energy_tf',
           u'instrumentalness_tf', u'key_tf', u'liveness_tf', u'loudness_tf',
           u'mode_tf', u'speechiness_tf', u'tempo_tf', u'time_signature_tf',
           u'valence_tf', 'lyric_char_len', 'lyric_word_len', 'verse_char_len_avg', 'verse_word_len_avg', 'n_verses']
    df_small = df[fields]
    df_norm = (df_small - df_small.mean()) / (df_small.max() - df_small.min())
    dv = DictVectorizer(sparse=False) 
    x = dv.fit_transform(df_norm.to_dict(orient='records'))
    imp = Imputer(missing_values=float('nan'), strategy='mean', axis=0)
    imp.fit(x)
    x_imp = imp.transform(x)
    return x_imp, y

#Fitting a random forest classifier
def rf_fit(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    param_grid = {'n_estimators': [90, 120, 200],
                      'max_depth': [3, 11, 15],
                      'min_samples_split': [10],
                      'min_samples_leaf': [1],
                      'max_features': ['log2']}
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, param_grid, cv=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    conf = confusion_matrix(y_pred, y_test)
    return (conf, clf)
    
#Fitting a logistic regression classifier
def lr_fit(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    conf = confusion_matrix(y_pred, y_test)
    return (conf, lr)

#Generating new features
def feature_engineering(df):
    df['datetime_album'] = pd.to_datetime(df.release_date_album, errors='coerce')
    df['album_year'] = df['datetime_album'].apply(lambda date: date.year)
    
    df['name_track_len'] = df.name_track.apply(lambda name: len(name.split()))
    
    df.Lyrics = df.Lyrics.fillna('')
    df['lyric_char_len'] = df.Lyrics.str.replace(" ", "").map(len)
    df['lyric_word_len'] = df.Lyrics.str.split().map(len)
    df['verse_char_len_avg'] = df.Lyrics.str.split('  ').map(lambda lyric: np.mean([len(line) for line in lyric]))
    df['verse_word_len_avg'] = df.Lyrics.str.split('  ').map(lambda lyric: np.mean([len(line.split()) for line in lyric]))
    df['n_verses'] = df.Lyrics.str.split('  ').map(len)
    return df
    
def main():
    lyrics_df = pd.read_csv('../data/Lyrics_enriched.csv')
    df = pd.read_csv('../data/Track_Features.csv')
    df = df.join(lyrics_df.set_index('track_id'), on = 'track_id')
    df = feature_engineering(df)
    x, y = get_xy(df)
    conf_lr, lr = lr_fit(x, y)
    conf_rf, clf = rf_fit(x, y)
    print "Logistic Regression confusion matrix:"
    print conf_lr
    print ""
    print "Random Forest confusion matrix:"
    print conf_rf

if __name__ == "__main__":
    main()