##### Imports #####

import dash 
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output 

import os 
import webbrowser
from threading import Timer


import pandas as pd , numpy as np 
import plotly.express as px 
import plotly.graph_objects as go

import base64
import io



##### Constants #####

df = pd.read_csv('Clean_delay_df.csv')

# For the connect histogram
connect = df[df['checkin_type'] == 'connect']
mean_connect = round(connect['delay_at_checkout_in_minutes'].mean())
median_connect = round(connect['delay_at_checkout_in_minutes'].median())
mode_connect = int(connect['delay_at_checkout_in_minutes'].mode())

# For the checkin histogram
mobile = df[df['checkin_type'] == 'mobile']
mean_mobile = round(mobile['delay_at_checkout_in_minutes'].mean())
median_mobile = round(mobile['delay_at_checkout_in_minutes'].median())
mode_mobile = int(mobile['delay_at_checkout_in_minutes'].mode())


# Png images 
with open("Percentage_of_lost_transaction.png", "rb") as image_file:
    Percentage_of_lost_transaction = base64.b64encode(image_file.read())

with open("ratio.png", "rb") as image_file:
    ratio_mobile = base64.b64encode(image_file.read())


def spaces(n=3): return html.Div([html.Br() for i in range(n)])

    


##### Figures #####

# Pie chart of the late versus not late variables 
labels = ['Late' , 'Early or on time']
values_ = [len(df[df['Late'] == 1]) , len(df[df['Late'] == 0])]
to_plot = pd.DataFrame({'labels' : labels , 'values_' : values_})
pie_fig = px.pie(to_plot , names='labels' , values=values_  , title='Percentage late versus early or on time')
pie_fig.update_layout(title=dict(x=0.5))  # Center horizontaly the titles

# Histogram of the delays for connect
hist_connect_fig = px.histogram(connect, x='delay_at_checkout_in_minutes',title=f'Delays in minutes for connect checkin : mean = {mean_connect} , median = {median_connect} , mode = {mode_connect} ', nbins=35)
hist_connect_fig.update_layout(title=dict(x=0.5)) 
hist_connect_fig.show()

# Histogram of the delays for mobile
hist_mobile_fig = px.histogram(mobile, x='delay_at_checkout_in_minutes',title=f'Delays in minutes for mobile checkin : mean = {mean_mobile} , median = {median_mobile} , mode = {mode_mobile} ', nbins=35)
hist_mobile_fig.update_layout(title=dict(x=0.5)) 
hist_mobile_fig.show()


##### Dashboard #####

# Instantiate the app 
app = dash.Dash(__name__)


# Main div
app.layout = html.Div(id='Main div' ,
                      
                      # Center everything
                      style={
                    'display': 'flex', # To align everything
                    'justify-content': 'center', # To center
                    'align-items': 'center', # To center
                    'flex-direction': 'column' }, # To display sequentialy 
                     
children=[

# Main title
html.H1(children=['------------------------------ New feature analysis ------------------------------'] , style={'border' : '1px solid black'}) ,



spaces() , # Add space


html.H2(children='Why do we need this feature anyway ?') ,

# Add the pie chart figure
dcc.Graph(figure=pie_fig) ,



spaces() , 

html.H2(children='Should we enable the feature for both types of checkin ?') , 


dcc.Graph(figure=hist_connect_fig) ,
dcc.Graph(figure=hist_mobile_fig) ,

spaces() ,

html.H2(children='The feature does not seems useful for the connect checkins ...'),
html.Img(src="data:image/png;base64,{}".format(Percentage_of_lost_transaction.decode())) , 

spaces() , 

html.H2(children='What would be the optimal minimum delay between rentals ?') ,
html.Img(src="data:image/png;base64,{}".format(ratio_mobile.decode())) , 


 ])







def open_browser():
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new('http://127.0.0.1:8050/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server()