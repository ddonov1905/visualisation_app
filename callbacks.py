    
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

#Create dataframe of all covid tests and interesting characteristics
heatmap_vars = ['Hematocrit', 'Eosinophils', 'Platelets',
                'Mean corpuscular hemoglobin (MCH)', 'Leukocytes',
                'Red blood cell distribution width (RDW)', 'Creatinine']

df_rw = covid[heatmap_vars].copy()
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


"""## Data processing and figure for bar chart cases per ward

"""
#Add ward column to covid df
wards = list()

for i in range(len(covid)):
    if covid['Patient addmited to semi-intensive unit (1=yes, 0=no)'][i] == 1:
        wards.append('Semi-intensive care')
    
    elif covid['Patient addmited to intensive care unit (1=yes, 0=no)'][i] == 1:
        wards.append('Intensive care')
    
    elif covid["Patient addmited to regular ward (1=yes, 0=no)"][i] == 1:
        wards.append('Regular Ward')
    
    else:
        wards.append(0)

covid['Ward'] = wards
    

#Make dataframe with the wards
df_wards = pd.DataFrame({'COVID':covid['SARS-Cov-2 exam result'], 
                         'Regular ward':covid['Patient addmited to regular ward (1=yes, 0=no)'],
                         'Semi-intensive care':covid['Patient addmited to semi-intensive unit (1=yes, 0=no)'],
                         'Intensive care':covid['Patient addmited to intensive care unit (1=yes, 0=no)']})

#Get dataframe with positive/negative cases and wards
df_pos = df_wards[df_wards['COVID']=='positive'].copy()
df_neg = df_wards[df_wards['COVID']=='negative'].copy()

#Get the percentages and result dataframe for pos cases
pos_reg = round(len(df_pos[df_pos['Regular ward']==1])/len(df_pos) * 100, 2)
pos_semi = round(len(df_pos[df_pos['Semi-intensive care']==1])/len(df_pos) * 100, 2)
pos_int = round(len(df_pos[df_pos['Intensive care']==1])/len(df_pos) * 100, 2)

#Get the percentages and result dataframe for neg cases
neg_reg = round(len(df_neg[df_neg['Regular ward']==1])/len(df_neg) * 100, 2)
neg_semi = round(len(df_neg[df_neg['Semi-intensive care']==1])/len(df_neg) * 100, 2)
neg_int = round(len(df_neg[df_neg['Intensive care']==1])/len(df_neg) * 100, 2)

#Get patient count for each ward and test result
count = []
i = ['Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)',
     'Patient addmited to intensive care unit (1=yes, 0=no)']*2
j = ['positive']*3 + ['negative']*3

for ward, result in zip(i,j):
    count.append(len(covid[(covid[ward]==1) & (covid['SARS-Cov-2 exam result']==result)]))

#Prepare dataframe for hospital ward bar chat
df_percs = pd.DataFrame({'wards': ['Regular Ward', 'Semi-intensive care', 'Intensive care',
                                   'Regular Ward', 'Semi-intensive care', 'Intensive care'],
                         'percs': [pos_reg, pos_semi, pos_int,
                                   neg_reg, neg_semi, neg_int],
                         'SARS-Cov-2 exam result':  ['positive', 'positive', 'positive',
                                   'negative', 'negative', 'negative'],
                         'count': count})
    
fig_2 = px.bar(df_percs, 
               title='COVID-19 distribution over hospital wards',
               x='wards',
               opacity=0.8,
               y='percs',
               text='percs',
               color='SARS-Cov-2 exam result',
               custom_data=['SARS-Cov-2 exam result', 'count'],
               category_orders={"SARS-Cov-2 exam result": ["Positive", "Negative"]},
               color_discrete_sequence=['#e36f10', '#2ba7cc'])
fig_2.update_layout(
    title_font=dict(color='#9D9D9D'),
    legend_font_color='white',
    legend_title_font_color='white',
    title_font_color="white",
    plot_bgcolor='#26232C',
    paper_bgcolor='#26232C',
    modebar_color = '#136d6d', #that's actually nice, I will leave it
    width=700,
    height=500,
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
    ),
    clickmode='event+select'
)
fig_2.update_traces(marker_line_width=0,
                    hovertemplate="Ward: "+"%{x}<br>Percents: %{y}\
                    <br>SARS-Cov-2 exam result: "+"%{customdata[0]}<br>Patient count: "+"%{customdata[1]} <extra></extra>")



"""## Data pre-processing Patient Age Quantile Postive COVID-19 case barchart"""

covid_positive = covid[covid['SARS-Cov-2 exam result']=='positive'].copy()

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

covid_negative = covid[covid['SARS-Cov-2 exam result']=='negative'].copy()

#number of people per age quartile who have had a positive COVID-19 result
naq_0 = round(100*len(covid_negative[covid_negative['Patient age quantile']==0])/len(covid[covid['Patient age quantile']==0]),1)
naq_1 = round(100*len(covid_negative[covid_negative['Patient age quantile']==1])/len(covid[covid['Patient age quantile']==1]),1)
naq_2 = round(100*len(covid_negative[covid_negative['Patient age quantile']==2])/len(covid[covid['Patient age quantile']==2]),1)
naq_3 = round(100*len(covid_negative[covid_negative['Patient age quantile']==3])/len(covid[covid['Patient age quantile']==3]),1)
naq_4 = round(100*len(covid_negative[covid_negative['Patient age quantile']==4])/len(covid[covid['Patient age quantile']==4]),1)
naq_5 = round(100*len(covid_negative[covid_negative['Patient age quantile']==5])/len(covid[covid['Patient age quantile']==5]),1)
naq_6 = round(100*len(covid_negative[covid_negative['Patient age quantile']==6])/len(covid[covid['Patient age quantile']==6]),1)
naq_7 = round(100*len(covid_negative[covid_negative['Patient age quantile']==7])/len(covid[covid['Patient age quantile']==7]),1)
naq_8 = round(100*len(covid_negative[covid_negative['Patient age quantile']==8])/len(covid[covid['Patient age quantile']==8]),1)
naq_9 = round(100*len(covid_negative[covid_negative['Patient age quantile']==9])/len(covid[covid['Patient age quantile']==9]),1)
naq_10 = round(100*len(covid_negative[covid_negative['Patient age quantile']==10])/len(covid[covid['Patient age quantile']==10]),1)
naq_11 = round(100*len(covid_negative[covid_negative['Patient age quantile']==11])/len(covid[covid['Patient age quantile']==11]),1)
naq_12 = round(100*len(covid_negative[covid_negative['Patient age quantile']==12])/len(covid[covid['Patient age quantile']==12]),1)
naq_13 = round(100*len(covid_negative[covid_negative['Patient age quantile']==13])/len(covid[covid['Patient age quantile']==13]),1)
naq_14 = round(100*len(covid_negative[covid_negative['Patient age quantile']==14])/len(covid[covid['Patient age quantile']==14]),1)
naq_15 = round(100*len(covid_negative[covid_negative['Patient age quantile']==15])/len(covid[covid['Patient age quantile']==15]),1)
naq_16 = round(100*len(covid_negative[covid_negative['Patient age quantile']==16])/len(covid[covid['Patient age quantile']==16]),1)
naq_17 = round(100*len(covid_negative[covid_negative['Patient age quantile']==17])/len(covid[covid['Patient age quantile']==17]),1)
naq_18 = round(100*len(covid_negative[covid_negative['Patient age quantile']==18])/len(covid[covid['Patient age quantile']==18]),1)
naq_19 = round(100*len(covid_negative[covid_negative['Patient age quantile']==19])/len(covid[covid['Patient age quantile']==19]),1)
naq_20 = round(100*len(covid_negative[covid_negative['Patient age quantile']==20]),1)

#sum over people per test and age quantile
quants = list(range(0,21))
count = []

for i in quants:
    count.append(len(covid_positive[covid_positive['Patient age quantile'] == i]))

for i in quants:
    count.append(len(covid_negative[covid_negative['Patient age quantile'] == i]))


#Create dataframe for the bar chart
df_ageq = pd.DataFrame({'quantile': ['0', '1','2','3','4','5','6','7','8','9','10','11','12',
                                     '13','14','15','16','17','18','19','20','0', '1','2','3',
                                     '4','5','6','7','8','9','10','11','12','13','14','15','16',
                                     '17','18','19','20'],
                         'percents': [paq_0, paq_1, paq_2, paq_3, paq_4, paq_5, paq_6, paq_7, paq_8,paq_9, paq_10,
                                      paq_11, paq_12, paq_13, paq_14, paq_15, paq_16, paq_17, paq_18, paq_19, paq_20,
                                      naq_0, naq_1, naq_2, naq_3, naq_4, naq_5, naq_6, naq_7, naq_8, naq_9, naq_10,
                                      naq_11, naq_12, naq_13, naq_14, naq_15, naq_16, naq_17, naq_18, naq_19, naq_20],
                         'SARS-Cov-2 exam result':  ['positive']*21 + ['negative']*21,
                         'count': count})

x = df_ageq['quantile']
y = df_ageq['percents']
    
fig_ageq = px.bar(df_ageq, 
                  x='quantile', 
                  y='percents', 
                  text='percents', 
                  title= 'Age quantile distibution of positive COVID-19 results', 
                  color='SARS-Cov-2 exam result',
                  color_discrete_sequence=['#e36f10', '#2ba7cc'],
                  category_orders={"SARS-Cov-2 exam result": ["Positive", "Negative"]},
                  custom_data=['SARS-Cov-2 exam result', 'count'],
                  opacity=0.8)
fig_ageq.update_layout(
    plot_bgcolor='#26232C',
    paper_bgcolor='#26232C',
    modebar_color = '#136d6d',
    title_font_color='white',
    legend_font_color='white',
    legend_title_font_color='white',
    width=750,
    height=500,
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
#         tickmode = 'array',
#         tickvals = [0, 5, 10, 15],
#         ticktext = ['0%', '5%', '10%', '15%'],
        showticklabels=True,
        showgrid =  False,
        gridcolor='#9D9D9D',
        title_font=dict(size=17, color='#9D9D9D'),
        gridwidth=1
    ),
    clickmode='event+select'
)
fig_ageq.update_xaxes(title='Age Quantile')
fig_ageq.update_traces(marker_line_width=0,
                       hovertemplate="Age quantile: "+"%{x}<br>Percents: %{y}\
                       <br>SARS-Cov-2 exam result: "+"%{customdata[0]}<br>Patient count: "+"%{customdata[1]} <extra></extra>")
fig_ageq.update_layout(barmode='stack', showlegend=True)



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


"""#Add column for Covid exam results (int)"""

covid['Covid-Exam-Result (int)'] = [0 if i == 'negative' 
                                    else 1 
                                    for i in covid.loc[:,'SARS-Cov-2 exam result']]




@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
               html.Div([
                        #Create drop down window for x-axis
                        html.Div([html.H4('Show on x-axis:'),
                                  dcc.Dropdown(
                                     id='xaxis-col',
                                     options=[{'label': i, 'value': i} for i in cont_vars],
                                     value='Hematocrit',
                                     clearable=False
                                 )], style={'width': '17%','text-align': 'center', 'position':'relative', 'left':80, 'display': 'inline-block'}),

                        #Create drop down window for y-axis
                        html.Div([html.H4('Show on y-axis:'),
                                  dcc.Dropdown(
                                     id='yaxis-col',
                                     options=[{'label': i, 'value': i} for i in cont_vars],
                                     value='Hemoglobin',
                                     clearable=False
                                 )], style={'width': '17%','text-align': 'center','position':'relative', 'left':150,'display': 'inline-block'}),
                     ],style={'padding': 20}),
    
                #Create graphic to show scatter plot
#                 html.Div([html.H2('Scatter plot for all continuous test variables', style={'text-align': 'left','position':'relative', 'left':150}),
#                           html.H2('title of boxplot', style={'position':'static', 'text-align': 'center'})]),
                
                html.Div([dcc.Graph(id='continuous-graphic',style={'width': '60%', 'float': 'left'}),
                          dcc.Graph(id='fig_box', style={'width': '40%', 'float': 'right'})]),

#                 html.Div([html.H2('title of barchart',style={'text-align': 'left','position':'relative', 'left':150}),
#                           html.H2('Distribution of positive cases over hospitalization', style={'text-align': 'right', 'position':'relative', 'left':50})]),
                
                html.Div([dcc.Graph(figure=fig_ageq, id='fig_ageq', style={'width': '45%', 'float': 'left', 'display': 'inline'}),
                          dcc.Graph(figure=fig_2, id='fig_wards',style={'float': 'right', 'display': 'inline'})]),
#                 html.Div([html.H2('blablablablbalblalbal',style={'text-align': 'left','position':'relative', 'left':150})]),
                html.Div([dcc.Graph(id='parallel-coord', style={'width': '100%','float': 'center','display': 'inline-block'})])
                

])
    elif tab == 'tab-2':
        return html.Div([html.H2('Heatmap of Historical Covid strains and mean '),
                         html.H2('Lymphocyte and mean Red Blood Cell values'),
                         html.Div([html.H4('Select COVID test:'),
                        dcc.Dropdown(
                            id='covid-type',
                            options=[{'label': i, 'value': i} for i in heatmap_cols],
                            value='Test result COVID'
                        )], style={'width': '48%', 'display': 'inline-block', 'float':'center'}),
                html.Div([dcc.Graph(id='covid-strain-heatmap', style={'width': '48%', 'display': 'inline-block', 'float': 'center'})],style={'padding': 30}),
                html.Div([html.Div([html.H2('Correlation heatmap for select patient characteristics'),
                                   #Create multi-select window for correlation heat map of all variables
                                   html.Div([html.H4('Select patient characteristics:'),
                                             dcc.Dropdown(id='corr-matrix',
                                                          options=[{'label': i, 'value': j} for i,j in zip(ht_cols, range(0, 74))],
                                                          value=list(range(0,24)),multi=True, 
                                                          style={'backgroundColor': '#26232C'})], style={'backgroundColor': '#26232C'})
                                   ]), 
                         dcc.Graph(id='correlation-heatmap-all-vars')])
])

    
    
    # ----------- Callbacks and Update function for scatterplot ----------
@app.callback(
    Output('continuous-graphic', 'figure'),
    [Input('xaxis-col', 'value'),
    Input('yaxis-col', 'value'),
    Input('fig_ageq', 'selectedData'),
    Input('fig_wards', 'selectedData')]
)

def update_graph(xaxis_col_name, yaxis_col_name, ageselect, wardselect):
    
    df = covid.copy()
    
    #age quantile loop
    if ageselect == None:
        pass
    elif ageselect["points"] == []:
        pass
    else:
        quantile = [i["x"] for i in ageselect["points"]]
        df = df[df['Patient age quantile'].isin(quantile)]
    
    #ward loop
    if wardselect == None:
        pass
    elif wardselect["points"] == []:
        pass
    
    else:
        ward = [i["x"] for i in wardselect["points"]]
        cov_result = [i["customdata"][0] for i in wardselect["points"]]
        df = df[(df['Ward'].isin(ward)) & (df["SARS-Cov-2 exam result"].isin(cov_result))]

    fig = px.scatter(df,
                     x=xaxis_col_name,
                     y=yaxis_col_name,
                     opacity=0.8,
                     title='Scatter plot for all continuous test variables',
                     color="SARS-Cov-2 exam result",
                     hover_name = df[['Patient ID', xaxis_col_name]]['Patient ID'],
                     color_discrete_sequence=['#2ba7cc','#e36f10'])
    
    fig.update_xaxes(title=xaxis_col_name)

    fig.update_yaxes(title=yaxis_col_name)
    
    fig.update_layout(plot_bgcolor='#26232C', modebar_color='#136d6d',
                      xaxis=dict(color='#9D9D9D',
                                 gridcolor='#9D9D9D'),
                      yaxis=dict(gridcolor='#9D9D9D',
                                 color="#9D9D9D"),
                      paper_bgcolor='#26232C',
                      legend_font_color='white',
                      legend_title_font_color='white',
                      title_font_color="white", 
                      margin={'l': 40, 'b': 40, 't': 40, 'r': 0}, 
                      hovermode='closest')


    
    fig.update_layout(clickmode='event+select')

    return fig

# Callbacks and Update function for boxplots
@app.callback(
    Output('fig_box', 'figure'),
    [Input('xaxis-col', 'value'),
    Input('yaxis-col', 'value'),
    Input('fig_ageq', 'selectedData'),
    Input('fig_wards', 'selectedData')]
)


def UpdateBoxplot(var1, var2, ageselect, wardselect):
    """
    Returns boxplot of selected variables in scatterplot
    Input: scatter dropdown variables
    Output: figure
    """
    df = covid.copy()
    
    #age quantile loop
    if ageselect == None:
        pass
    elif ageselect["points"] == []:
        pass
    else:
        quantile = [i["x"] for i in ageselect["points"]]
        df = df[df['Patient age quantile'].isin(quantile)]
    
    #ward loop
    if wardselect == None:
        pass
    elif wardselect["points"] == []:
        pass
    else:
        ward = [i["x"] for i in wardselect["points"]]
        df = df[df['Ward'].isin(ward)]

    fig = px.box(df,
                 y=[var1, var2],
                 title='Summary statistics',
                 color="SARS-Cov-2 exam result",
                 color_discrete_sequence=['#2ba7cc','#e36f10'])
    
    fig.update_layout(
        plot_bgcolor='#26232C',        
        title_font_color="white",
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

# ---------- Callbacks and Update function for heatmap ----------
@app.callback(
    Output('covid-strain-heatmap', 'figure'),
    [Input('covid-type', 'value')]
)

def update_heatmap(covid_type):
    
    means = [[df_rw[(df_rw[covid_type] == 'positive')]['Hematocrit'].mean(),
              df_rw[(df_rw[covid_type] == 'positive')]['Eosinophils'].mean(),
              df_rw[(df_rw[covid_type] == 'positive')]['Platelets'].mean(),
             df_rw[(df_rw[covid_type] == 'positive')]['Mean corpuscular hemoglobin (MCH)'].mean(),
             df_rw[(df_rw[covid_type] == 'positive')]['Leukocytes'].mean(),
             df_rw[(df_rw[covid_type] == 'positive')]['Red blood cell distribution width (RDW)'].mean(),
             df_rw[(df_rw[covid_type] == 'positive')]['Creatinine'].mean()],
             [df_rw[(df_rw[covid_type] == 'negative')]['Eosinophils'].mean(),
              df_rw[(df_rw[covid_type] == 'negative')]['Hematocrit'].mean(),
              df_rw[(df_rw[covid_type] == 'negative')]['Platelets'].mean(),
             df_rw[(df_rw[covid_type] == 'negative')]['Mean corpuscular hemoglobin (MCH)'].mean(),
             df_rw[(df_rw[covid_type] == 'negative')]['Leukocytes'].mean(),
             df_rw[(df_rw[covid_type] == 'negative')]['Red blood cell distribution width (RDW)'].mean(),
             df_rw[(df_rw[covid_type] == 'negative')]['Creatinine'].mean()]
            ]
    
    fig = px.imshow(means, x=heatmap_vars, y=['Positive', 'Negative'],
                   labels=dict(x="Patient Recording", y=f"{covid_type}", color="Mean cell recording"),
                   color_continuous_scale='Viridis')

    fig.update_layout(plot_bgcolor='#26232C',paper_bgcolor='#26232C', modebar_color = '#136d6d' ,title='',
                       xaxis=dict(color='#9D9D9D'),
                       yaxis={'title': f'{covid_type}', 'color':'#9D9D9D'})
    
    fig.layout.coloraxis.colorbar.title='Mean test recording'
    fig.layout.coloraxis.colorbar.title.font.color='#9D9D9D'
    fig.layout.coloraxis.colorbar.tickfont.color='#9D9D9D'
    fig.layout.coloraxis.colorbar.x=1.1
    
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
                       colorbar = dict(title=' Correlation', title_font_color='#9D9D9D',tickfont_color='#9D9D9D'),
                       colorscale = 'Viridis')
    
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

# ---------- Update function for Parallel Coordinates Plot ---------
@app.callback(
    Output('parallel-coord', 'figure'),
    [Input('fig_ageq', 'selectedData'),
     Input('fig_wards', 'selectedData')]
)

def UpdatePCP(ageselect, wardselect):
        
    df = covid.copy()
    
    #age quantile loop
    if ageselect == None:
        pass
    elif ageselect["points"] == []:
        pass
    else:
        quantile = [i["x"] for i in ageselect["points"]]
        df = df[df['Patient age quantile'].isin(quantile)]
    
    #ward loop
    if wardselect == None:
        pass
    elif wardselect["points"] == []:
        pass
    else:
        ward = [i["x"] for i in wardselect["points"]]
        cov_result = [i["customdata"][0] for i in wardselect["points"]]
        df = df[(df['Ward'].isin(ward)) & (df["SARS-Cov-2 exam result"].isin(cov_result))]
    
    fig = px.parallel_coordinates(df,
                              dimensions=['Hematocrit', 'Eosinophils', 'Platelets',
                                          'Mean corpuscular hemoglobin (MCH)', 'Leukocytes',
                                          'Red blood cell distribution width (RDW)', 'Creatinine'],
                              color='Covid-Exam-Result (int)',
                              color_continuous_scale=['#2BA7CC','#e36f10'],
                              range_color=(0,1))
    
    fig.update_layout(plot_bgcolor='#26232C', modebar_color = '#136d6d',
                      xaxis=dict(color="#9D9D9D",
                                 gridcolor='#9D9D9D',
                                 title_font=dict(color='#9D9D9D')),
                      yaxis=dict(gridcolor="#9D9D9D",
                                 color="#9D9D9D",
                                 title_font=dict(color='#9D9D9D')),
                      legend=dict(bgcolor='#9D9D9D'),
                      title_font=dict(color='#9D9D9D'),
                      legend_font_color='#9D9D9D',
                      paper_bgcolor='#26232C',
                      margin={'l': 40, 'b': 40, 't': 40, 'r': 0}, 
                      hovermode='closest')
    
    fig.update_traces(labelfont=dict(color='#9D9D9D'), selector=dict(type='parcoords'))
    fig.update_traces(line_colorbar_tickcolor='#9D9D9D',selector=dict(type='parcoords'))
    fig.update_traces(line_colorbar_title_font_color='#9D9D9D',selector=dict(type='parcoords'))
    fig.update_traces(rangefont_color='#9D9D9D',selector=dict(type='parcoords'))
    
    fig.layout.coloraxis.colorbar.title='SARS-Cov-2 exam result'
    fig.layout.coloraxis.colorbar.title.font.color='#9D9D9D'
    fig.layout.coloraxis.colorbar.showticklabels=False
    
    return fig
    
#blablablablablablalb
