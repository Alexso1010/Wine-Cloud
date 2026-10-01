import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import naive_bayes,svm
from sklearn.svm import SVC
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, classification_report
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


df = pd.read_csv('data/madrid_meteo_data.csv')

df.loc[df[' Events'].isnull(), ' Events'] = 'Sun' # Les valeurs manquantes correspondent à des journées ensoleillées

# Est ce que toutes les valeurs manquantes sont des soleils ?


X = df.drop(columns=['CET', 'WindDirDegrees'], axis=1)

# Pour les rafales de vent, on remplace les valeurs manquantes par 0 (pas de rafales)
X[' Max Gust SpeedKm/h'] = X[' Max Gust SpeedKm/h'].fillna(0)

# CHOIX 1 supprimer les lignes avec des valeurs manquantes (sauf pour Max Gust SpeedKm/h)
# ET supprimer les lignes avec des y trop rares (moins de 10 occurrences)

X.dropna(inplace=True) 

etiquettes_drop = X[' Events'].value_counts()[X[' Events'].value_counts() < 10].index
X.drop(X[X[' Events'].isin(etiquettes_drop)].index, inplace=True)
Y = X[' Events']
X.drop(columns=[' Events'], inplace=True)    # 5400 lignes gardées sur 6800

# 70% apprentissage, 30% temporaire
X_app, X_temp, Y_app, Y_temp = train_test_split(
    X, Y,
    test_size=0.30,
    random_state=42,
    shuffle=True)

# Diviser les 30% restants en 15% validation et 15% test
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp,
    test_size=0.50,
    random_state=42,
    shuffle=True)


print ( 'nombre d occurrences de chaque classe : \n', Y.value_counts(normalize=True))
le = LabelEncoder()

Y_app_encoded = le.fit_transform(Y_app)
Y_val_encoded = le.transform(Y_val)
Y_test_encoded = le.transform(Y_test)

weights = compute_sample_weight(
    class_weight="balanced",
    y=Y_app_encoded
)

mod  =xgb.XGBClassifier()
mod.fit(X_app, Y_app_encoded, sample_weight=weights)

Y_pred = mod.predict(X_test)

print(classification_report(
        Y_test_encoded,
        Y_pred,
        target_names=le.classes_))

ConfusionMatrixDisplay.from_predictions(
    Y_test_encoded,
    Y_pred,
    display_labels=le.classes_,
    xticks_rotation=90)

plt.show()