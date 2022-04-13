# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:51:55 2022

@author: 91798
"""

import numpy as np
import pickle as pk
import streamlit as st

loaded_model=pk.load(open('C:/Users/91798/Desktop/data/diabetes_system.sav','rb'))

# creating function

def diabetes_prediction(input_data):
   
    input_data_as_numpy_data=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_data.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if prediction==0:
      return "The person is not diabetic"
    else:
      return "The person is diabetic"    


def main():
    # giving title
    st.title('Diabetes Prediction App')
    
    # getting input from user
    
    Pregnancies=st.text_input("No of Pregnancies")
    Glucose=st.text_input("What is the glucose level?")
    BloodPressure=st.text_input("What is the Blood pressure level?")
    SkinThickness=st.text_input("What is Skin Thickness value?")
    Insulin=st.text_input("What is the Insulin level")
    BMI=st.text_input("What is BMI?")
    DiabetesPedigreeFunction=st.text_input("What is the Diabetes Pedigree Function value")
    Age=st.text_input("What is Age?")
    
    # code for Prediction
    diagnosis=" "
    if st.button('Diagnosis Test Results'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
    
if __name__ == "__main__":
    main()
    
    