import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


iris = load_iris()
iris_x = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=.2, random_state=1)

log = LogisticRegression(class_weight='balanced')
log.fit(X_train, y_train)


def predictor(input_data):
	input_data_as_numpy_array = np.asarray(input_data)
	reshaped_numpy_array = input_data_as_numpy_array.reshape(1,-1)
	
	prediction = log.predict(reshaped_numpy_array)
	return prediction
	
def main():
    st.title('Iris Prediction')
    
    sepal_length = st.text_input('sepal length (cm)')
    sepal_width = st.text_input('sepal width (cm)')
    petal_length = st.text_input('petal length (cm)')
    petal_width = st.text_input('petal width (cm)')
    
    if st.button('Iris Prediction Result'):
        prediction = predictor([float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)])
        st.success(prediction)

if __name__=="__main__":
	main()
