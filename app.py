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
                   ui.h5('Edit this table to show what questions measure each skill', ui.br(), '(questions can measure more than one skill)'),
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
    ui.input_file('load', 'Load in your dataset'),
    ui.input_select('type_model',
                    'Choose a Model Type:',
                    {'dino': 'Has at least one skill measured by each question',
                     'dina': 'Has all skills measured by each question'}
                    # 'lcdm': 'LCDM'},
    ),
    ui.input_slider('attr_num', 'Number of skills in your assessment', 2, 5, 2),
    ui.input_slider('att1_alpha', 'Beta Distribution - Skill 1: Alpha', 0, 50, 1, step = .5),
    ui.input_slider('att1_beta', 'Beta Distribution - Skill 1: Beta', 0, 50, 1, step = .5),
    ui.input_checkbox("all_same_prior", "Keep all the same skill priors as above", True),
    ui.panel_conditional(
      '!input.all_same_prior',
      ui.input_slider('att2_alpha', 'Beta Distribution - Skill 2: Alpha', 0, 50, 1, step = .5),
      ui.input_slider('att2_beta', 'Beta Distribution - Skill 2: Beta', 0, 50, 1, step = .5),
      ui.input_slider('att3_alpha', 'Beta Distribution - Skill 3: Alpha', 0, 50, 1, step = .5),
      ui.input_slider('att3_beta', 'Beta Distribution - Skill 3: Beta', 0, 50, 1, step = .5),
      ui.input_slider('att4_alpha', 'Beta Distribution - Skill 4: Alpha', 0, 50, 1, step = .5),
      ui.input_slider('att4_beta', 'Beta Distribution - Skill 4: Beta', 0, 50, 1, step = .5),
      ui.input_slider('att5_alpha', 'Beta Distribution - Skill 5: Alpha', 0, 50, 1, step = .5),
      ui.input_slider('att5_beta', 'Beta Distribution - Skill 5: Beta', 0, 50, 1, step = .5),
    ),
    ui.h6('How likely students are to have the skill, but get question incorrect'),
    ui.input_slider('slip_alpha', 'Beta Distribution - Slip: Alpha', 0, 50, 5, step = .5),
    ui.input_slider('slip_beta', 'Beta Distribution - Slip: Beta', 0, 50, 20, step = .5),
    ui.h6('How likely students are to not have the skill, but get question correct by guessing'),
    ui.input_slider('guess_alpha', 'Beta Distribution - Guess: Alpha', 0, 50, 5, step = .5),
    ui.input_slider('guess_beta', 'Beta Distribution - Guess: Beta', 0, 50, 20, step = .5),
    ui.input_action_button('build_model', 'Update parameters'),
    ui.input_action_button('run_model', 'Run model'),
    ui.download_button('download_q', 'Download Q-Matrix (CSV)'),
    ui.input_checkbox('use_intial_values', 'Check this box if model does not converge', False)
    # ui.input_text_area('priors', 'Include Priors for Attributes', rows = 6),
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
        'value': np.random.beta(alpha, beta, size = 1000)
    })
    
    plot = (
      pn.ggplot(dist_data,
                pn.aes('value'))
      + pn.geom_density(color = 'white',
                        fill = 'seagreen')
      + pn.scale_x_continuous(limits = [0, 1],
                              breaks = np.arange(0, 1.1, .1))
      + pn.labs(title = 'How likely is it that students have the skill',
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
  
  # THIS NEEDS WORK
  @reactive.event(input.build_model)
  def get_inits():
    n = input.attr_num()
    df = loaded_data()
    alpha = create_alpha()
    
    slip_alpha = input.slip_alpha()
    slip_beta = input.slip_beta()
    guess_alpha = input.guess_alpha()
    guess_beta = input.guess_beta()
    
    slip = np.random.beta(slip_alpha, slip_beta, size = 1000)
    guess = np.random.beta(guess_alpha, guess_beta, size = 1000)
    
    slip_start = np.mean(slip)
    guess_start = np.mean(guess)
    
    slip_sd = np.std(slip)
    guess_sd = np.std(guess)
    
    if n == 2:
      if input.use_intial_values() == True:
        alpha1 = input.att1_alpha()
        beta1 = input.att1_beta()
        alpha2 = input.att2_alpha()
        beta2 = input.att2_beta()
        
        att1_dist = np.random.beta(alpha1, beta1, size = 1000)
        att2_dist = np.random.beta(alpha2, beta2, size = 1000)
        
        start = np.mean([att1_dist, att2_dist])
        sd = np.std([att1_dist, att2_dist])
        
        return {
          'nu': np.repeat(1/alpha.shape[0], alpha.shape[0]),
          'slip': np.clip(np.random.uniform((slip_start - slip_sd), (slip_start + slip_sd), size = df.shape[1]).tolist(), 0, 1),
          'guess': np.clip(np.random.uniform((guess_start - guess_sd), (guess_start + guess_sd), size = df.shape[1]).tolist(), 0, 1),
          'lambda1': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda2': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1)
      }
      else:
        return None
    
    elif n == 3:
      if input.use_intial_values() == True:
        alpha1 = input.att1_alpha()
        beta1 = input.att1_beta()
        alpha2 = input.att2_alpha()
        beta2 = input.att2_beta()
        alpha3 = input.att3_alpha()
        beta3 = input.att3_beta()
        
        att1_dist = np.random.beta(alpha1, beta1, size = 1000)
        att2_dist = np.random.beta(alpha2, beta2, size = 1000)
        att3_dist = np.random.beta(alpha3, beta3, size = 1000)
        
        start = np.mean([att1_dist, att2_dist, att3_dist])
        sd = np.std([att1_dist, att2_dist, att3_dist])
        
        return {
          'nu': np.repeat(1/alpha.shape[0], alpha.shape[0]),
          'slip': np.clip(np.random.uniform((slip_start - slip_sd), (slip_start + slip_sd), size = df.shape[1]).tolist(), 0, 1),
          'guess': np.clip(np.random.uniform((guess_start - guess_sd), (guess_start + guess_sd), size = df.shape[1]).tolist(), 0, 1),
          'lambda1': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda2': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda3': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1)
      }
      else:
        return None
    
    elif n == 4:
      if input.use_intial_values() == True:
        alpha1 = input.att1_alpha()
        beta1 = input.att1_beta()
        alpha2 = input.att2_alpha()
        beta2 = input.att2_beta()
        alpha3 = input.att3_alpha()
        beta3 = input.att3_beta()
        alpha4 = input.att4_alpha()
        beta4 = input.att4_beta()
        
        att1_dist = np.random.beta(alpha1, beta1, size = 1000)
        att2_dist = np.random.beta(alpha2, beta2, size = 1000)
        att3_dist = np.random.beta(alpha3, beta3, size = 1000)
        att4_dist = np.random.beta(alpha4, beta4, size = 1000)
        
        start = np.mean([att1_dist, att2_dist, att3_dist, att4_dist])
        sd = np.std([att1_dist, att2_dist, att3_dist, att4_dist])
        
        return {
          'nu': np.repeat(1/alpha.shape[0], alpha.shape[0]),
          'slip': np.clip(np.random.uniform((slip_start - slip_sd), (slip_start + slip_sd), size = df.shape[1]).tolist(), 0, 1),
          'guess': np.clip(np.random.uniform((guess_start - guess_sd), (guess_start + guess_sd), size = df.shape[1]).tolist(), 0, 1),
          'lambda1': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda2': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda3': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda4': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1)
      }
      else:
        return None

    elif n == 5:
      if input.use_initial_values() == True:
        alpha1 = input.att1_alpha()
        beta1 = input.att1_beta()
        alpha2 = input.att2_alpha()
        beta2 = input.att2_beta()
        alpha3 = input.att3_alpha()
        beta3 = input.att3_beta()
        alpha4 = input.att4_alpha()
        beta4 = input.att4_beta()
        alpha5 = input.att5_alpha()
        beta5 = input.att5_beta()
        
        att1_dist = np.random.beta(alpha1, beta1, size = 1000)
        att2_dist = np.random.beta(alpha2, beta2, size = 1000)
        att3_dist = np.random.beta(alpha3, beta3, size = 1000)
        att4_dist = np.random.beta(alpha4, beta4, size = 1000)
        att5_dist = np.random.beta(alpha5, beta5, size = 1000)
        
        start = np.mean([att1_dist, att2_dist, att3_dist, att4_dist, att5_dist])
        sd = np.std([att1_dist, att2_dist, att3_dist, att4_dist, att5_dist])
        
        return {
          'nu': np.repeat(1/alpha.shape[0], alpha.shape[0]),
          'slip': np.clip(np.random.uniform((slip_start - slip_sd), (slip_start + slip_sd), size = df.shape[1]).tolist(), 0, 1),
          'guess': np.clip(np.random.uniform((guess_start - guess_sd), (guess_start + guess_sd), size = df.shape[1]).tolist(), 0, 1),
          'lambda1': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda2': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda3': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda4': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1),
          'lambda5': np.clip(np.random.uniform((start - sd), (start + sd)), 0, 1)
      }
      else:
        return None
    

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
    q = empty_q.get()
    if q is None or q.empty:
      raise SafeException('Q-matrix is empty. Load data and edit the table first.')
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