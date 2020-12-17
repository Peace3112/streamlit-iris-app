import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
df = pd.read_csv('iris.csv')
df.set_index('Id',inplace=True)
X = df.drop('Species',axis=1)
y = df['Species']
st.title('Iris Flower Prediction')
st.write('*This app predicts the type of Iris flower based on the sepal length and width, and petal length and width.*')

image= Image.open(r"C:\Users\ADEGBITE BOLA\Pictures\iris.png")
st.sidebar.image(image, caption= 'Iris Flowers', width=300 )
st.sidebar.text('Please choose:')
sepal_length= st.sidebar.slider('Sepal Length (in cm)', 4.3, 7.9, 5.1)
sepal_width= st.sidebar.slider('Sepal Width (in cm)', 2.0, 4.4, 3.5)
petal_length=st.sidebar.slider('Petal Length (in cm)', 1.0, 6.9, 1.4)
petal_width= st.sidebar.slider('Petal Width (in cm)', 0.1, 2.5, 0.2)

data=pd.DataFrame({'sepal_length':sepal_length,
'sepal_width':sepal_width,
'petal_length':petal_length,
'petal_width':petal_width},index=[0])

show= st.checkbox('Show user Input parameters', (0,1))
if show==1:
    st.table(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier(min_weight_fraction_leaf=0.3, criterion='entropy', max_leaf_nodes=4)
dt.fit(X_train,y_train)
predict= dt.predict(data)

st.subheader('Prediction')
for val in predict:
    i=val
st.success('The flower is {}'. format(str(i)))
name=i.split('-')[1].lower()
st.image(Image.open(r"C:\Users\ADEGBITE BOLA\Pictures\iris-{}.png".format(name)), width=210)





