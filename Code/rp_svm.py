# Fuente: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# %matplotlib inline
def loadObjectsFile(file):
    myfile = open(file)
    data_myfile = myfile.read()
    myfile.close()
    return data_myfile

objpointers = loadObjectsFile("breast-cancer.testors.obj")
objpointers = objpointers.split()

#Quitar la O
for i in range(0,len(objpointers)):
    objpointers[i] = int(re.sub('\D', '', objpointers[i]))
objpointers.sort()

bankdata = pd.read_csv("breast-cancer.data") # Cambiar aqui, si se altera el archivo

train_df_cols = []

for col in bankdata.columns:
    train_df_cols.append(col)

train_df = pd.DataFrame(columns=train_df_cols)

c = 1
for i in objpointers:
    train_df.loc[c] = bankdata.iloc[i]
    c = c+1

print(bankdata.shape)

print(bankdata.head())

# Preproceso de labels a int, usar solo si es necesario
le = preprocessing.LabelEncoder()
for column_name in bankdata.columns:
    if bankdata[column_name].dtype == object:
        bankdata[column_name] = le.fit_transform(bankdata[column_name])
    else:
        pass

for column_name in train_df.columns:
    if train_df[column_name].dtype == object:
        train_df[column_name] = le.fit_transform(train_df[column_name])
    else:
        pass

Xt = train_df.drop('class', axis=1)
yt = train_df["class"]

X = bankdata.drop('class', axis=1)
y = bankdata['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(Xt, yt)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#validacion cruzada validacion entre metodos con validacion cruzada (Olvera) revisar o si no pedirselo, pedir datos de tesis