
from app import app
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction
import plotly.express as px
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns


"""import data"""

covid = pd.read_excel("dataset.xlsx")

"""## Data pre-processing for historical covid-strain heatmap"""

#Create dataframe of all covid tests and Lymphocyte and Red blood cell tests
df_rw = covid.iloc[:, 10:12].copy()
df_rw['Test result COVID'] = covid['SARS-Cov-2 exam result']
df_rw['Result CoronavirusNL63'] = covid['CoronavirusNL63']
df_rw['Result CoronavirusHKU1'] = covid['Coronavirus HKU1']
df_rw['Result Coronavirus229E'] = covid['Coronavirus229E']
df_rw['Result CoronavirusOC43'] = covid['CoronavirusOC43']

#Remove all Null values
df_rw.dropna(inplace=True)

#Change str values to negative/positive for historical covid strains
my_cols =['Result CoronavirusNL63', 'Result CoronavirusHKU1', 'Result Coronavirus229E', 'Result CoronavirusOC43']

for col in my_cols:
    df_rw.loc[df_rw[col]=='not_detected', col] = 'negative' 
    
for col in my_cols:
    df_rw.loc[df_rw[col]=='detected', col] = 'positive'
    
heatmap_cols = ['Test result COVID', 'Result CoronavirusNL63',
                'Result CoronavirusHKU1', 'Result Coronavirus229E', 'Result CoronavirusOC43']

"""## Data processing and figure for bar chart positive cases per ward

"""

#Make dataframe with the wards
df_wards = pd.DataFrame({'COVID':covid['SARS-Cov-2 exam result'], 
                         'Regular ward':covid['Patient addmited to regular ward (1=yes, 0=no)'],
                         'Semi-intensive care':covid['Patient addmited to semi-intensive unit (1=yes, 0=no)'],
                         'Intensive care':covid['Patient addmited to intensive care unit (1=yes, 0=no)']})

#Get dataframe with positive cases and wards
df_pos = df_wards[df_wards['COVID']=='positive'].copy()

#Get the percentages and result dataframe
pos_reg = round(len(df_pos[df_pos['Regular ward']==1])/len(df_pos) * 100, 2)
pos_semi = round(len(df_pos[df_pos['Semi-intensive care']==1])/len(df_pos) * 100, 2)
pos_int = round(len(df_pos[df_pos['Intensive care']==1])/len(df_pos) * 100, 2)

df_percs = pd.DataFrame({'wards': ['Regular Ward', 'Semi-intensive care', 'Intensive care'],
                         'percs': [pos_reg, pos_semi, pos_int]})

x = df_percs['wards']
y = df_percs['percs']
    
fig_2 = px.bar(df_percs, x='wards', y='percs', text='percs', color_discrete_sequence=['#0983b0','#42B0D6','#C52828'])
fig_2.update_layout(
    title_font=dict(color='#9D9D9D'),
    plot_bgcolor='#26232C',
    paper_bgcolor='#26232C',
    modebar_color = '#136d6d', #that's actually nice, I will leave it
    width=1100,
    height=600,
    xaxis = dict(
        title = '',
        color='#9D9D9D',
        tickfont_size=14),
    yaxis=dict(
        title = 'Percentual cases',
        color="#9D9D9D",
        titlefont_size=16,
        tickfont_size=14,
        tickmode = 'array',
        tickvals = [2, 4, 6],
        ticktext = ['2%', '4%', '6%'],
        showgrid =  True,
        gridcolor='#9D9D9D',
        title_font=dict(size=17, color='#9D9D9D')
    )
)
fig_2.update_traces(marker_line_width=0)

"""## Data pre-processing Patient Age Quantile Postive COVID-19 case barchart"""

covid_positive = covid[covid['SARS-Cov-2 exam result']=='positive']

#number of people per age quartile who have had a positive COVID-19 result
paq_0 = round(100*len(covid_positive[covid_positive['Patient age quantile']==0])/len(covid[covid['Patient age quantile']==0]),1)
paq_1 = round(100*len(covid_positive[covid_positive['Patient age quantile']==1])/len(covid[covid['Patient age quantile']==1]),1)
paq_2 = round(100*len(covid_positive[covid_positive['Patient age quantile']==2])/len(covid[covid['Patient age quantile']==2]),1)
paq_3 = round(100*len(covid_positive[covid_positive['Patient age quantile']==3])/len(covid[covid['Patient age quantile']==3]),1)
paq_4 = round(100*len(covid_positive[covid_positive['Patient age quantile']==4])/len(covid[covid['Patient age quantile']==4]),1)
paq_5 = round(100*len(covid_positive[covid_positive['Patient age quantile']==5])/len(covid[covid['Patient age quantile']==5]),1)
paq_6 = round(100*len(covid_positive[covid_positive['Patient age quantile']==6])/len(covid[covid['Patient age quantile']==6]),1)
paq_7 = round(100*len(covid_positive[covid_positive['Patient age quantile']==7])/len(covid[covid['Patient age quantile']==7]),1)
paq_8 = round(100*len(covid_positive[covid_positive['Patient age quantile']==8])/len(covid[covid['Patient age quantile']==8]),1)
paq_9 = round(100*len(covid_positive[covid_positive['Patient age quantile']==9])/len(covid[covid['Patient age quantile']==9]),1)
paq_10 = round(100*len(covid_positive[covid_positive['Patient age quantile']==10])/len(covid[covid['Patient age quantile']==10]),1)
paq_11 = round(100*len(covid_positive[covid_positive['Patient age quantile']==11])/len(covid[covid['Patient age quantile']==11]),1)
paq_12 = round(100*len(covid_positive[covid_positive['Patient age quantile']==12])/len(covid[covid['Patient age quantile']==12]),1)
paq_13 = round(100*len(covid_positive[covid_positive['Patient age quantile']==13])/len(covid[covid['Patient age quantile']==13]),1)
paq_14 = round(100*len(covid_positive[covid_positive['Patient age quantile']==14])/len(covid[covid['Patient age quantile']==14]),1)
paq_15 = round(100*len(covid_positive[covid_positive['Patient age quantile']==15])/len(covid[covid['Patient age quantile']==15]),1)
paq_16 = round(100*len(covid_positive[covid_positive['Patient age quantile']==16])/len(covid[covid['Patient age quantile']==16]),1)
paq_17 = round(100*len(covid_positive[covid_positive['Patient age quantile']==17])/len(covid[covid['Patient age quantile']==17]),1)
paq_18 = round(100*len(covid_positive[covid_positive['Patient age quantile']==18])/len(covid[covid['Patient age quantile']==18]),1)
paq_19 = round(100*len(covid_positive[covid_positive['Patient age quantile']==19])/len(covid[covid['Patient age quantile']==19]),1)
paq_20 = round(100*len(covid_positive[covid_positive['Patient age quantile']==20]),1)


df_ageq = pd.DataFrame({'quantile': ['0', '1','2','3','4','5','6','7','8','9','10','11','12',
                                     '13','14','15','16','17','18','19','20'],
                         'percents': [paq_0, paq_1, paq_2, paq_3, paq_4, paq_5, paq_6, paq_7, paq_8,paq_9, paq_10,
                                      paq_11, paq_12, paq_13, paq_14, paq_15, paq_16, paq_17, paq_18, paq_19, paq_20]})

x = df_ageq['quantile']
y = df_ageq['percents']
    
fig_ageq = px.bar(df_ageq, x='quantile', y='percents', text='percents', color_discrete_sequence=['#0983b0','#42B0D6','#C52828'])
fig_ageq.update_layout(
#     title='Age quantile distibution of positive COVID-19 results',
    plot_bgcolor='#26232C',
    paper_bgcolor='#26232C',
    modebar_color = '#136d6d', #that's actually nice, I will leave it
    width=700,
    height=430,
    xaxis = dict(
        title = '',
        color="#9D9D9D",
        tickfont_size=14,
        title_font=dict(size=20, color='#9D9D9D')),
    yaxis=dict(
        title = 'Percentual cases',
        color="#9D9D9D",
        titlefont_size=16,
        tickfont_size=14,
        tickmode = 'array',
        tickvals = [0, 5, 10, 15],
        ticktext = ['0%', '5%', '10%', '15%'],
        showticklabels=False,
        showgrid =  False,
        gridcolor='#9D9D9D',
        title_font=dict(size=17, color='#9D9D9D'),
        gridwidth=1
    ),
    clickmode='select'
)
fig_ageq.update_xaxes(title='Age Quantile')
fig_ageq.update_traces(marker_line_width=0)
fig_ageq.update_layout(barmode='overlay', showlegend=True)

"""## Data pre-processing for correlation heatmap for all patient characteristics"""

df_corr = covid #Create new df to store variables useful for correlation computations

#Traverse each column and delete if more than 95% of values is Null
for i, j in zip(df_corr.isnull().sum().index, df_corr.isnull().sum()):
    if j >= 5362:
        df_corr.drop(i, axis=1, inplace=True)
        
    else:
        continue
#5644 patients and 46 attributes left

#Extract all correlation coeficients per index
corr = [[i for i in j] for j in df_corr.corr().iloc]

#Extract all column names
ht_cols = np.array(list(df_corr.corr().columns))

"""### Create list of all continuous variables in dataset"""

#List comprehension to create list of all quantitative variables
cont_vars = [i for i in covid.columns[:112] if covid[i].dtype == 'float64']



@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([html.Div([html.H2('Scatter plot for all continuous test variables'),
                        #Create drop down window for x-axis
                        html.Div([html.H4('Show on x-axis:'),
                                  dcc.Dropdown(
                                     id='xaxis-col',
                                     options=[{'label': i, 'value': i} for i in cont_vars],
                                     value='Hematocrit',
                                     clearable=False
                                 )], style={'width': '17%','text-align': 'center', 'position':'relative', 'left':420, 'display': 'inline-block'}),

                        #Create drop down window for y-axis
                        html.Div([html.H4('Show on y-axis:'),
                                  dcc.Dropdown(
                                     id='yaxis-col',
                                     options=[{'label': i, 'value': i} for i in cont_vars],
                                     value='Hemoglobin',
                                     clearable=False
                                 )], style={'width': '17%','text-align': 'center','position':'relative', 'left':540,'display': 'inline-block'}),
                     ]),
    
                #Create graphic to show scatter plot
                dcc.Graph(id='continuous-graphic'),

                html.Div([html.Div(
                    [html.H2('Bar chart showing people age quantile distibution of all COVID-19 results'),
                    html.H2('and summary statistic boxplots of the characteristics of the selected group')]),
                    html.Div([dcc.Graph(id='fig_ageq', style={'width': '25%', 'float': 'left', 'display': 'inline-block'}),
                              dcc.Graph(id='fig_box', style={'width': '45%', 'float': 'right', 'display': 'inline-block'})]), 
                ])
        ])
    elif tab == 'tab-2':
        return html.Div([html.H2('Heatmap of Historical Covid strains and mean '),
                         html.H2('Lymphocyte and mean Red Blood Cell values'),
                         html.Div([html.H4('Select COVID test:'),
                        dcc.Dropdown(
                            id='covid-type',
                            options=[{'label': i, 'value': i} for i in heatmap_cols],
                            value='Test result COVID'
                        )], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([dcc.Graph(id='covid-redbloodcell-lymphocytes-heatmap')],
                          style={'width': '48%', 'display': 'inline-block', 'float': 'left'})
])

    
    elif tab == 'tab-3':
        return html.Div([html.Div([html.H2('Distribution of positive cases over hospitalization')]),
              
               html.Div([dcc.Graph(figure=fig_2)],
                  style={'width': '48%','position':'relative', 'left':230, 'display': 'inline-block'}),
        ])
    elif tab == 'tab-4':
        return html.Div([html.Div([html.H2('Bar chart showing people age quantile distibution of all COVID-19 results')]),
              
                         html.Div([dcc.Graph(figure=fig_ageq)],
                          style={'width': '48%', 'position':'relative', 'left':220,'display': 'inline-block'}),
        ])
    elif tab == 'tab-5':
        return html.Div([html.Div([html.H2('Correlation heatmap for select patient characteristics'),
                                   #Create multi-select window for correlation heat map of all variables
                                   html.Div([html.H4('Select patient characteristics:'),
                                             dcc.Dropdown(id='corr-matrix',
                                                          options=[{'label': i, 'value': j} for i,j in zip(ht_cols, range(0, 74))],
                                                          value=list(range(0,24)),multi=True)], style={})
                                  ]), 
                         dcc.Graph(id='correlation-heatmap-all-vars')],style={})
    elif tab == 'tab-info':
        return html.Div([html.Div([html.H2('info info info'),
                                   #Create multi-select window for correlation heat map of all variables
                                   html.Div([html.H4('Select patient characteristics:'),
                                             dcc.Dropdown(id='corr-matrix',
                                                          options=[{'label': i, 'value': j} for i,j in zip(ht_cols, range(0, 74))],
                                                          value=list(range(0,24)),multi=True)], style={})
                                  ]), 
                         dcc.Graph(id='correlation-heatmap-all-vars')],style={})

# ----------- Callbacks and Update function for scatterplot ----------
@app.callback(
    Output('continuous-graphic', 'figure'),
    [Input('xaxis-col', 'value'),
    Input('yaxis-col', 'value')]
)

def update_graph(xaxis_col_name, yaxis_col_name):

    fig = px.scatter(covid,
                     x=xaxis_col_name,
                     y=yaxis_col_name,
                     color=covid["SARS-Cov-2 exam result"],
                     hover_name = covid[['Patient ID', xaxis_col_name]]['Patient ID'],
                     color_discrete_sequence=['#0983b0','#ffa600','#C52828'])

    fig.update_layout(plot_bgcolor='#26232C', modebar_color = '#136d6d',
                      xaxis=dict(color='#9D9D9D',
                                 gridcolor='#9D9D9D'),
                      yaxis=dict(gridcolor='#9D9D9D',
                                 color="#9D9D9D"),
                      legend_font_color='white',
                      legend_title_font_color='white',
                      paper_bgcolor='#26232C', 
                      margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, 
                      hovermode='closest')

    fig.update_xaxes(title=xaxis_col_name)

    fig.update_yaxes(title=yaxis_col_name)

    fig.update_layout(clickmode='event+select')

    return fig

# Callbacks and Update function for boxplots
@app.callback(
    Output('fig_box', 'figure'),
    [Input('xaxis-col', 'value'),
    Input('yaxis-col', 'value')]
)


def UpdateBoxplot(var1, var2):
    """
    Returns boxplot of selected variables in scatterplot
    Input: scatter dropdown variables
    Output: figure
    """
    fig = px.box(covid, y=[var1, var2], color="SARS-Cov-2 exam result")
    
    fig.update_layout(
        plot_bgcolor='#26232C',
        paper_bgcolor='#26232C',
        modebar_color = '#136d6d',
        xaxis=dict(color='#9D9D9D',
                   gridcolor='#9D9D9D'),
        yaxis=dict(gridcolor='#9D9D9D',
                   color="#9D9D9D"),
        legend_font_color='white',
        legend_title_font_color='white',
    )
    
    return fig

# Callbacks and Update function for bar chart
@app.callback(
    Output('fig_ageq', 'figure'),
    [Input('continuous-graphic', 'clickData')]
)

def update_age(clickData):
    
    if clickData == None:
        return fig_ageq
    
    else:
        
        fig = go.Figure(fig_ageq)
        df=covid

        click_id = clickData['points'][0]["hovertext"]
        age = df[df['Patient ID']==click_id].iloc[0]['Patient age quantile']
        perc = df_ageq['percents'][age]
    
        fig.add_bar(name='Patient ID: '+str(click_id), y=[perc], x=[age], marker=dict(color="LightSeaGreen"),
                   alignmentgroup='True', texttemplate="%{y:.f}", textposition="inside", textfont=dict(color="#ffffff"),
                   hovertemplate="Patient: "+str(click_id)+"<br>Age quantile: "+"%{x}<br>Percents: %{y} <extra></extra>")
        return fig

# ---------- Callbacks and Update function for heatmap ----------
@app.callback(
    Output('covid-redbloodcell-lymphocytes-heatmap', 'figure'),
    [Input('covid-type', 'value')]
)

def update_heatmap(covid_type):
    
    x = [[df_rw[(df_rw[covid_type] == 'positive')]['Red blood Cells'].mean(),
         df_rw[(df_rw[covid_type] == 'negative')]['Red blood Cells'].mean()],
         [df_rw[(df_rw[covid_type] == 'positive')]['Lymphocytes'].mean(),
         df_rw[(df_rw[covid_type] == 'negative')]['Lymphocytes'].mean()]
    ]
    
    fig = px.imshow(x, x=['Red Blood Cells', 'Lymphocytes'], y=['Positive', 'Negative'],
                   labels=dict(x="Cell Type", y=f"{covid_type}", color="Mean cell recording"))

    fig.update_layout(coloraxis=dict(colorscale='Viridis'), plot_bgcolor='#26232C',paper_bgcolor='#26232C', modebar_color = '#136d6d' ,title='',
                       xaxis=dict(title='Cell type',
                                  color='#9D9D9D'),
                       yaxis={'title': f'{covid_type}', 'color':'#9D9D9D'})
    
    return fig


# ---------- Callbacks and Update function for correlation heatmap ---------
@app.callback(
    Output('correlation-heatmap-all-vars', 'figure'),
    [Input('corr-matrix', 'value'),
     Input('corr-matrix', 'value')]
)

def update_corr_matrix(selection, selection_2):
    if len(selection) == 11:
        selection[0] = selection[10]
        del selection[10]
    
    htmap = go.Heatmap(z = corr, x = ht_cols[selection],
                       y = ht_cols[selection_2],
                       hovertemplate='Y: %{y}<br>X: %{x}<br>Correlation: %{z}<extra></extra>',
                       colorbar = dict(title=' Correlation'), colorscale = 'Viridis'
                      )
    
    layout = go.Layout(plot_bgcolor='#26232C',
                       paper_bgcolor='#26232C',
                       modebar_color = '#136d6d', 
                       height=900,
                       yaxis=dict(color='#9D9D9D'),
                       xaxis=dict(color='#9D9D9D'))
    
    fig = go.Figure(data=[htmap], layout=layout)

    fig.update_layout(
    updatemenus=[
        dict(
            buttons=list([
                dict(
                    args=["colorscale", "Viridis"],
                    label="Viridis",
                    method="restyle"
                ),
                dict(
                    args=["colorscale", "Cividis"],
                    label="Cividis",
                    method="restyle"
                )
            ]),
            type = "buttons",
            direction="right",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ])

    return fig
