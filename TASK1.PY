import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import cufflinks as cf
import plotly.offline as pyo
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode,plot,iplot
# from google.colab import drive
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# %matplotlib inline



#Mounting Drive - I USED IT IN MY GOOGLE COLAB
# drive.mount('/content/gdrive')
# iris=pd.read_csv('/content/gdrive/MyDrive/Iris.csv')
iris=pd.read_csv('Iris.csv')



print(iris)
iris.drop("Id",axis=1,inplace=True) #here "inplace=true" is used to make changes permanently to variable
print(iris)



#vizualization
scatter=px.scatter(iris,x="Species",y="PetalWidthCm",size="PetalWidthCm")
scatter.show()
bar=px.bar(iris,x="Species",y="PetalWidthCm")
bar.show()
line=px.line(iris,x="Species",y="PetalWidthCm")
line.show()
scatter_matrix=px.scatter_matrix(iris,color='Species',title='Iris',dimensions=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])
scatter_matrix.show()



#Data Processing
x=iris.drop("Species",axis=1)
print(x)
y=iris.Species
print(y)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)
x=np.array(x)
print(x)



#Dividing into training and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
print(xtest)



#Training the model using Logistic Regression
lr=LogisticRegression()
lr.fit(xtrain,ytrain)



#Prediction
prediction=lr.predict(xtest)
prediction = [1 if x >= 0.5 else 0 for x in prediction]
accuracy=accuracy_score(ytest,prediction)*100



#Accuracy percentage
print("Accuracy for LogisticRegression : ",accuracy)
print(ytest)
print(np.array(prediction))



#Comparing Predicted data with Testing data
array1=ytest
array2=prediction
fig = px.line(title="Line Graph TestData vs prediction")
fig.add_scatter(x=[i for i in range(len(array1))], y=array1, mode="lines", line=dict(color="blue"), showlegend=True, name="TestData")
fig.add_scatter(x=[i for i in range(len(array2))], y=array2, mode="lines", line=dict(color="red"), showlegend=True, name="Prediction")
fig.show()


# The mismatch between the actual test data and the predicted data on the graph indicates that there is an error in the prediction.
# Here accuracy is very low so we will be using decision tree classifier to improve accuracy



#Using Decision Tree Classifier to increace accuracy
dt=tree.DecisionTreeClassifier()
dt.fit(xtrain,ytrain)



#Prediction
prediction=dt.predict(xtest)
accuracy=accuracy_score(ytest,prediction)*100



#Accuracy percentage
print("Accuracy for DesisionTreeClassifier : ",accuracy)
print(ytest)
print(np.array(prediction))



#Comparing Predicted data with Testing data
array1=ytest
array2=prediction
fig = px.line(title="Line Graph TestData vs prediction")
fig.add_scatter(x=[i for i in range(len(array1))], y=array1, mode="lines", line=dict(color="blue"), showlegend=True, name="TestData")
fig.add_scatter(x=[i for i in range(len(array2))], y=array2, mode="lines", line=dict(color="red"), showlegend=True, name="Prediction")
fig.show()