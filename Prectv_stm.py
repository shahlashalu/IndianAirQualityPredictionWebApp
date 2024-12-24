# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 12:26:22 2024

@author: hp
"""

import numpy as np
import pickle
    

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (4.7,7.5,10.9,45.9)

# changing the input data to numpy array
input_data_as_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 0):
  print('Good')
elif (prediction[0] == 1):
  print('Moderate')
elif (prediction[0] == 2):
  print('Satisfacory')
elif (prediction[0] == 3):
  print('Poor')
elif (prediction[0] == 4):
  print('Hazardous')
elif (prediction[0] == 5):
  print('Severe')