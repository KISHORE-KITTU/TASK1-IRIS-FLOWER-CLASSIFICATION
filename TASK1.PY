import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import cufflinks as cf
import plotly.offline as pyo
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode,plot,iplot
from google.colab import drive
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# %matplotlib inline



drive.mount('/content/gdrive')
iris=pd.read_csv('/content/gdrive/MyDrive/Iris.csv') #Mounting Drive
iris
iris.drop("Id",axis=1,inplace=True) #here "inplace=true" is used to make changes permanently to variable
iris



#vizualization
px.scatter(iris,x="Species",y="PetalWidthCm",size="PetalWidthCm")
px.bar(iris,x="Species",y="PetalWidthCm")
px.line(iris,x="Species",y="PetalWidthCm")
px.scatter_matrix(iris,color='Species',title='Iris',dimensions=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])



#Data Processing
x=iris.drop("Species",axis=1)
x
y=iris.Species
y
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
y
x=np.array(x)
x



#Dividing into training and testing data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)
xtest



#Training the model using Linear Regression
lr=LinearRegression()
lr.fit(xtrain,ytrain)



#Prediction
prediction=lr.predict(xtest)
prediction = [1 if x >= 0.5 else 0 for x in prediction]
accuracy=accuracy_score(ytest,prediction)*100



#Accuracy percentage
accuracy
print(ytest)
print(prediction)



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
accuracy
print(ytest)
print(prediction)



#Comparing Predicted data with Testing data
array1=ytest
array2=prediction
fig = px.line(title="Line Graph TestData vs prediction")
fig.add_scatter(x=[i for i in range(len(array1))], y=array1, mode="lines", line=dict(color="blue"), showlegend=True, name="TestData")
fig.add_scatter(x=[i for i in range(len(array2))], y=array2, mode="lines", line=dict(color="red"), showlegend=True, name="Prediction")
fig.show()
