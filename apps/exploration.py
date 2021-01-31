
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] #css style template

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

layout = html.Div([
    html.Div([html.H1('COVID-19 Visualizer - Group 27')]),
    html.Div([
        dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Scatter Plot',
                value='tab-1',
                selected_className='custom-tab-1',
                style={'background-color': '#006a68',
                       'border-radius': '20px',
                       'color':'white',
                       'border': '0px'}),
        dcc.Tab(label='Correlation Heat Maps',
                value='tab-2',
                selected_className='custom-tab-2',
                style={'background-color': '#006a68',
                       'border-radius': '20px',
                       'color':'white',
                       'border': '0px'})]),
        html.Div(id='tabs-example-content')]),
    
])

