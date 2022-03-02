
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.p

data = pd.read_csv(r"D:\AAdityAA\Python_Sandbox\diabetes-200520-125813.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

mean = np.mean(data['DiabetesPedigreeFunction'])
# print(mean)
std = np.std(data['DiabetesPedigreeFunction'])
# print(std)

t = 3

out = []

for i in data['DiabetesPedigreeFunction']:
  z = (i-mean)/std
  if np.abs(z)>t:
    out.append(i)

print(out)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=10)

dt_model = DecisionTreeClassifier()
if dt_model.fit:
  print('DT MODEL OUTPUT')
  dt_model.fit(X_train,y_train)



dt_prediction = dt_model.predict(X_test)
print('Decision Tree accuracy = ', metrics.accuracy_score(dt_prediction,y_test))

sns.pairplot(data)

plt.show()



