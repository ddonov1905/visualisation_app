import dash_html_components as html
import dash_bootstrap_components as dbc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the COVID-19 Visualization tool of group 8", className="text-center")
                    , className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H5(children='bla bla bla bla'
                                     )
                    , className="mb-4")
            ]),

        dbc.Row([
            dbc.Col(html.H5(children='It consists of two main pages: Home, this is an introduction page to the Group 8 visualization tool'
                                     'Exploration, which gives the oppurtunity to exploral several bla bla bla bla')
                    , className="mb-5")
        ]),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H3(children='Get the original datasets used in this project',
                                               className="text-center"),
                                       dbc.Button("Covid-19",
                                                  href="https://www.kaggle.com/einsteindata4u/covid19",
                                                  color="primary",
                                                  className="mt-3")
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4"),

            dbc.Col(dbc.Card(children=[html.H3(children='You can find the code for this project in',
                                               className="text-center"),
                                       dbc.Button("GitHub",
                                                  href="https://github.com",
                                                  color="primary",
                                                  className="mt-3"),
                                       ],
                             body=True, color="dark", outline=True)
                    , width=6, className="mb-4")
        ], className="mb-5")

    ])

])