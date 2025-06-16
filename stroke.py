import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd

data = pd.read_csv("stroke.csv")
collums = ["id","stroke"]
data = data.dropna()
Y = data["stroke"]
stroke_data = data.drop(collums,axis=1)
stroke_data = stroke_data.dropna()

lb = LabelEncoder()
categories_list = ["gender","ever_married","work_type","Residence_type","smoking_status"]

stroke_data_lb = stroke_data.copy()
for category in categories_list:
    stroke_data_lb[category] = lb.fit_transform(stroke_data[category].astype(str))
    
X_train,X_test,y_train,y_test = train_test_split(stroke_data_lb,Y,test_size=0.3,shuffle=True,random_state=10)

knn_pipeline = Pipeline(steps=[
  ("normalizacao", MinMaxScaler()),  
  ("KNN", KNeighborsClassifier(n_neighbors=3))
])

knn_pipeline.fit(X_train, y_train)
y_pred = knn_pipeline.predict(X_test)
y_pred_prob = knn_pipeline.predict_proba(X_test)

knn = KNeighborsClassifier(n_neighbors=3)

y_pred = knn_pipeline.predict(X_test)
y_pred_prob = knn_pipeline.predict_proba(X_test)
print(f"Acurácia de treinamento: {knn_pipeline.score(X_train, y_train)}")

target_names = ["no stroke","stroke"]
relatorio = classification_report(y_test, y_pred, target_names=target_names)
print("Relatório de classificação:")
print(relatorio)

mat_conf = confusion_matrix(y_test, y_pred)
print("Matriz de confusão:")
print(mat_conf)
