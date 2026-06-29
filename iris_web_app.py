import streamlit as st
import pandas as pd
import numpy as np
import pickle

loaded_model = pickle.load(open('iris_prediction.pkl', 'rb'))

def predictor(input_data):
	input_data_as_numpy_array = np.asarray(input_data)
	reshaped_numpy_array = input_data_as_numpy_array.reshape(1,-1)
	
	prediction = loaded_model.predict(reshaped_numpy_array)
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