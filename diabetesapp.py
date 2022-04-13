# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:30:56 2022

@author: 91798
"""

import numpy as np
import pickle as pk
from sklearn.preprocessing import StandardScaler





loaded_model=pk.load(open('C:/Users/91798/Desktop/data/diabetes_system.sav','rb'))
input_data=(2,90,80,14,55,24.4,0.249,24)
input_data_as_numpy_data=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_data.reshape(1,-1)
# input_data_reshaped_std_data=scaler.transform(input_data_reshaped)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
