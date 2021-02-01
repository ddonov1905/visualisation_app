import dash_html_components as html
import dash_bootstrap_components as dbc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the COVID-19 Visualization tool of group 27", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='A dataset about the Diagnosis of COVID-19 and its clinical spectrum has been analysied in this web tool. The link for the dataset you can find below.'
                                     )
                    , className="mb-4")
            ]),

        dbc.Row([
            dbc.Col(html.H5(children='The tool consists of two main pages: Home tab, this is an introduction page to the Group 27 visualization tool. Exploration tab, which gives the oppurtunity to explore the dataset and find interesting patterns')
                    , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Get the original datasets used in this project',
                                               className="text-center"),
                                       dbc.Button("Covid-19",
                                                  href="https://www.kaggle.com/einsteindata4u/covid19",
                                                  color="primary",
                                                  target="_blank",
                                                  className="mt-3")
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='You can find the code for this project in',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com/ddonov1905/visualisation_app/",
                                                  color="primary",
                                                  target="_blank",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4")
        ], className="mb-5")

    ])

])
