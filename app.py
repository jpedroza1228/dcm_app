import pandas as pd
import matplotlib
import numpy as np
import plotnine as pn
from great_tables import GT
from janitor import clean_names
from pyhere import here
import arviz as az
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import SafeException
import shinyswatch

pd.set_option('display.max_columns', None)
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

# --------------------------------------------------------------------------------------------------------

# code
def q_lower(x):
    return x.quantile(.025)
  
def q_upper(x):
    return x.quantile(.975)

def acceptable_fit_stat(inference_data, func_name = ['waic', 'loo']):
  if func_name == 'waic':
    est = np.abs(az.waic(inference_data).iloc[0])
    se = az.waic(inference_data).iloc[1]
    
    if est > se * 2.5:
      print('Absolute difference is greater than 2.5 x the standard error of the difference. Model is acceptable.')
      
    else:
      print('Absolute difference is not greater than 2.5 x the standard error of the difference. Model is not acceptable.')
  elif func_name == 'loo':
    est = np.abs(az.loo(inference_data).iloc[0])
    se = az.loo(inference_data).iloc[1]
    
    if est > se * 2.5:
      print('Absolute difference is greater than 2.5 x the standard error of the difference. Model is acceptable.')
      
    else:
      print('Absolute difference is not greater than 2.5 x the standard error of the difference. Model is not acceptable.')

# --------------------------------------------------------------------------------------------------------

app_ui = ui.page_navbar(  
  ui.nav_panel('Examining Data',
               ui.layout_columns(
                 ui.page_fluid(
                   ui.h5('Top 5 rows of dataset'),
                   ui.output_data_frame('dataset'),
                   ui.output_plot('att1_dist')
                   ),
                 ui.page_fluid(
                   ui.h5('Q-Matrix (Editable)'),
                   ui.output_data_frame('qmatrix')
                   ),
                 col_widths = (5, 7)
                 )
               ), 
  ui.nav_spacer(),
  ui.nav_control(ui.input_dark_mode()),
  # ui.nav_panel('B', 'Page B content'),  
  # ui.nav_panel('C', 'Page C content'),  
  title = 'DCMs For Practitioners',  
  id = 'page',
  sidebar = ui.sidebar(
    ui.input_file('load', 'Read in your dataset'),
    ui.input_select('type_model',
                    'Choose a Model Type:',
                    {'dino': 'DINO'}
       #'dina': 'DINA', 'lcdm': 'LCDM'},
    ),
    ui.input_slider('attr_num', 'Choose number of attributes', 2, 5, 2),
    ui.input_slider('att1_alpha', 'Beta Distribution - Attribute 1: Alpha', 0, 50, 1, step = .5),
    ui.input_slider('att1_beta', 'Beta Distribution - Attribute 1: Beta', 0, 50, 1, step = .5),
    # ui.input_text_area('priors', 'Include Priors for Attributes', rows = 6),
    ui.input_action_button('build_model', 'Update parameters'),
    ui.input_action_button('run_model', 'Run'),
    ui.download_button("download_q", "Download Q-Matrix (CSV)") # remove this after knowing q-matrix is accurate
  ),
  theme = shinyswatch.theme.flatly
)

# --------------------------------------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):
  empty_q = reactive.Value(pd.DataFrame())
  
  pass

  @reactive.calc
  def loaded_data():
    uploaded = input.load()
    if not uploaded:
        return None
    df = pd.read_csv(uploaded[0]['datapath'])
    cols = [
        col for col in df.columns
        if set(df[col].dropna().unique()).issubset({0, 1})
    ]
    return df[cols]

  @render.data_frame
  def dataset():
    df = loaded_data()
    if df is None:
        return None
    return render.DataTable(df.head(),
                            height = '225px')
    
  @reactive.effect
  def _initialize_qmatrix():
    df = loaded_data()
    if df is not None:
      n_attrs = input.attr_num()
      attr_cols = [f'A{i}' for i in range(1, n_attrs + 1)]
      # Build the fresh grid
      q = pd.DataFrame(0, index=df.columns, columns=attr_cols).reset_index()
      q.rename(columns={"index": "Item"}, inplace=True)
      empty_q.set(q)
  
  @reactive.effect
  @reactive.event(input.qmatrix_cell_edit)
  def _update_qmatrix():
    edit = input.qmatrix_cell_edit()
    q = empty_q.get().copy()
    q.iat[edit["row"], edit["col"]] = edit["value"]
    empty_q.set(q)
    
  @render.data_frame
  def qmatrix():
      return render.DataTable(empty_q.get(),
                              editable = True)

  # @render.data_frame
  # def qmatrix():
  #   df = loaded_data()
  #   if df is None:
  #       return None
  #   n_attrs = input.attr_num()
  #   attr_cols = [f'A{i}' for i in range(1, n_attrs + 1)]
  #   q = pd.DataFrame(0, index=df.columns, columns = attr_cols)
  #   q.index.name = 'Item'
  #   return render.DataTable(q.reset_index(),
  #                           editable = True)
  
  @render.text
  def att1_avalue():
    return f'{input.att1_alpha()}'
  
  @render.text
  def att1_bvalue():
    return f'{input.att1_beta()}'
  
  @render.plot
  @reactive.event(input.build_model)
  def att1_dist():
    alpha = input.att1_alpha()
    beta = input.att1_beta()
    
    if alpha <= 0 or beta <= 0:
        return None
    
    dist_data = pd.DataFrame({
        'value': np.random.beta(alpha, beta, size=500)
    })
    
    plot = (
      pn.ggplot(dist_data,
                pn.aes('value'))
      + pn.geom_density(color = 'white',
                        fill = 'seagreen')
      + pn.scale_x_continuous(limits = [0, 1],
                              breaks = np.arange(0, 1.1, .1))
      + pn.labs(title = 'What Percentage of Respondents Should Have the Skill',
                x = 'Probability',
                y = '')
      + pn.theme_light()
    )
    return plot.draw()
  
  @reactive.effect
  def create_alpha():
    n = input.attr_num()
    
    if n == 2:
      alpha = pd.DataFrame([(a, b) for a in np.arange(2) for b in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'A1',
                                      1: 'A2'})
    elif n == 3:
      alpha = pd.DataFrame([(a, b, c) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'A1',
                                      1: 'A2',
                                      2: 'A3'})
      
    elif n == 4:
      alpha = pd.DataFrame([(a, b, c, d) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'A1',
                                      1: 'A2',
                                      2: 'A3',
                                      3: 'A4'})
    elif n == 5:
      alpha = pd.DataFrame([(a, b, c, d, e) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2) for e in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'A1',
                                      1: 'A2',
                                      2: 'A3',
                                      3: 'A4',
                                      4: 'A5'})
    return alpha
  
  # @reactive.effect
  # def _update_priors():
  #   n = input.attr_num()
  #   priors = '\n'.join(f'lambda{i} ~ beta(1,1);' for i in range(1, n + 1))
  #   ui.update_text_area('priors',
  #                       value = priors)
  
  # THIS NEEDS WORK
  # @reactive.event(input.build_model)
  # def get_inits():
  #   df = loaded_data()
    
  #   return {
  #       "nu": np.repeat(1/stan_dict['C'], stan_dict['C']),  # Start with equal class probabilities
  #       "slip": np.random.uniform(0.05, 0.15, size = stan_dict['I']).tolist(),
  #       "guess": np.random.uniform(0.05, 0.15, size = stan_dict['I']).tolist(),
  #       "lambda1": np.random.uniform(0.7, 0.9),
  #       "lambda2": np.random.uniform(0.7, 0.9),
  #       "lambda3": np.random.uniform(0.7, 0.9)
  #   }

  @reactive.event(input.build_model)
  def update_model():
    return f'{input.build_model()}'
      
  @render.text
  @reactive.event(input.run_model)
  def model_button():
    return f'{input.run_model()}'
  
  # everything below this
  @render.download(filename = 'q_matrix_updated.csv')
  def download_q():
    # Get the current state of the edited Q-matrix
    q = qmatrix.get()
    # Use a generator to yield the CSV data
    if q is not None:
        yield q.to_csv(index = False)

# --------------------------------------------------------------------------------------------------------

app = App(app_ui, server)

# --------------------------------------------------------------------------------------------------------

# from shiny import App, ui, render, reactive, Inputs, Outputs, Session

# # ... (UI code remains mostly the same)

# def server(input: Inputs, output: Outputs, session: Session):
    
#     @reactive.effect
#     def _update_priors():
#         n = input.attr_num()
#         # Vectorized format is cleaner: lambda ~ beta(1,1);
#         # But we can also generate individual ones if the user wants specific control
#         priors = '\n'.join(f'lambda[{i}] ~ beta(1, 1);' for i in range(1, n + 1))
#         ui.update_text_area('priors', value=priors)

#     @reactive.calc
#     def construct_stan_model():
#         # This takes the template above and injects the UI inputs
#         # You would store the "Generalized Stan Template" in a string variable
#         template = """... (The Stan code from above) ..."""
        
#         user_priors = input.priors()
#         full_model = template.replace("[PRIORS_PLACEHOLDER]", user_priors)
#         return full_model

#     @render.text
#     @reactive.event(input.build_model)
#     def model_preview():
#         # This shows the user what the generated Stan code looks like
#         return construct_stan_model()

#     @reactive.event(input.run_model)
#     def run_stan():
#         model_string = construct_stan_model()
#         # Here you would pass model_string to cmdstanpy or pystan
#         print("Running model with attributes:", input.attr_num())
#         # result = sm.sample(data=my_data_dict)