# Api and data manipulation
from typing import Union , List
import pandas as pd , numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import gunicorn
import json

# Preprocess and predict
from catboost import CatBoostRegressor
from sklearn.preprocessing import QuantileTransformer



# Write a little description
description = """
# Pricing optimisation api
## endpoints : /predict

predict : post endpoint that uses a AI model to returns the optimum prices.
Input data should be in json format and should contains lists of the predictor values in that order : 
model_key	mileage	engine_power	fuel	paint_color	car_type	private_parking_available	has_gps	has_air_conditioning	automatic_car	has_getaround_connect	has_speed_regulator	winter_tires 
"""

# Instantiate the app
app = FastAPI(title='get_around_api' , description=description)


# Create a class to properly receive the data
class Observation(BaseModel):
    
    model_key : str 
    mileage : int 
    engine_power : int
    fuel : str
    paint_color : str
    car_type : str
    private_parking_available : bool
    has_gps : bool
    has_air_conditioning : bool
    automatic_car : bool
    has_getaround_connect : bool
    has_speed_regulator : bool
    winter_tires : bool

# Create a second class to recevie lists of observations
class ObservationList(BaseModel):
    observations: List[Observation]


# The endpoint
@app.post("/predict")
async def predict(obs_lists : ObservationList):
    ''' Use an AI model to predict the optimum prices for one or several observations. 
        The data should be sent in the form : {"observations": [{"model_key": "Peugeot", "mileage": 50000, "engine_power": 150, "fuel": "essence", "paint_color": "noir", "car_type": "citadine", "private_parking_available": True, "has_gps": True, "has_air_conditioning": True, "automatic_car": False, "has_getaround_connect": False, "has_speed_regulator": True, "winter_tires": False}, {"model_key": "DEF456", "mileage": 75000, "engine_power": 120, "fuel": "diesel", "paint_color": "rouge", "car_type": "break", "private_parking_available": False, "has_gps": False, "has_air_conditioning": True, "automatic_car": True, "has_getaround_connect": True, "has_speed_regulator": False, "winter_tires": True}]}
        Example with python : 
        
        import requests 

        params = {'observations' : [{'model_key' : 'Citroën' , 'mileage' : 140411 , 'engine_power' : 100 , 'fuel' : 'diesel' , 'paint_color' : 'black' ,
                                    'car_type' : 'convertible' , 'private_parking_available' : True , 'has_gps' : True , 'has_air_conditioning' : True ,
                                    'automatic_car' : False , 'has_getaround_connect' : True , 'has_speed_regulator' : True ,
                                    'winter_tires' : True}]}

        url = 'http://localhost:8000/predict'

        response = requests.post(url=url , json=params)

        print(response.json())
    '''

    ##### Constants #####
    df_target = pd.read_csv('get_around_pricing_project.csv' , index_col='Unnamed: 0')
    model = CatBoostRegressor(verbose=0)
    model.load_model('model.cbm') # My trained and optimized model

    # The target encoded dictionaries
    target_encoding_values = {}
    for col in df_target.select_dtypes(include=object):
        mean_values = dict(df_target.groupby(col)['rental_price_per_day'].mean())
        target_encoding_values[col] = mean_values


    # Data processing
    to_predict = []
    for obs in obs_lists.observations:
        to_predict.append({
            'model_key': obs.model_key,
            'mileage': obs.mileage,
            'engine_power': obs.engine_power,
            'fuel': obs.fuel,
            'paint_color': obs.paint_color,
            'car_type': obs.car_type,
            'private_parking_available': obs.private_parking_available,
            'has_gps': obs.has_gps,
            'has_air_conditioning': obs.has_air_conditioning,
            'automatic_car': obs.automatic_car,
            'has_getaround_connect': obs.has_getaround_connect,
            'has_speed_regulator': obs.has_speed_regulator,
            'winter_tires': obs.winter_tires
        })
    to_predict = pd.DataFrame(to_predict)

    # Numerical features
    num_features = ['mileage', 'engine_power']
    for col in num_features:
        to_predict[col] = QuantileTransformer().fit_transform(to_predict[col].values.reshape(-1, 1))

    # Categorical features
    for col in to_predict.select_dtypes(include=object):
        mean_values = target_encoding_values.get(col)
        to_predict[col] = [mean_values.get(val) for val in list(to_predict[col])]
        to_predict[col] = QuantileTransformer().fit_transform(to_predict[col].values.reshape(-1, 1))

    # Make the prediction
    predictions = list(model.predict(to_predict))

    # Return the predictions in json format
    return json.dumps({'predictions': predictions})




