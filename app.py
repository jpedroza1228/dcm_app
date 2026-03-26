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
from pathlib import Path
from cmdstanpy import CmdStanModel

pd.set_option('display.max_columns', None)
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

# --------------------------------------------------------------------------------------------------------

# code
def stan_datachunk():
  data = 'data {\nint<lower=1> J;\nint<lower=1> I;\nint<lower=1> C;\nint<lower=1> K;\nmatrix<lower=0,upper=1> [J,I] Y;\nmatrix<lower=0,upper=1> [I,K] Q;\nmatrix<lower=0,upper=1> [C,K] alpha;\n}'
  return data

print(stan_datachunk())

def stan_paramchunk():
  param = 'parameters {\nordered[C] raw_nu_ordered;\nvector<lower=0, upper=1>[I] slip;\nvector<lower=0, upper=1>[I] guess;\n'
  return param
    
print(stan_paramchunk())

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
  ui.nav_panel('Background Information',
               ui.layout_columns(
                 ui.page_fluid(
                   ui.markdown("""
                               This page has all the information required to use the following pages for using diagnostic classification models (DCM) to see which respondents have the skills measured in assessments. Currently, this application is focused on DCMs for smaller samples as an introduction to using DCMs as an preventive measure to flag respondents that may not have a grasp of skills assessed in an assessment.
                             
                               *This application should not be used as the sole measure of assessing proficiency in respondents. These models are not perfect, especially for smaller samples, so user judgment should be used to determine respondent proficiency.*
                             
                               An example of a Q-matrix is shown below.
                               """),
                   ui.tags.table(
                    ui.tags.thead(
                      ui.tags.tr(
                        ui.tags.th('Item'),
                        ui.tags.th('A1'),
                        ui.tags.th('A2')
                        )
                      ),
                   ui.tags.tbody(
                     ui.tags.tr(
                       ui.tags.td('Item1'),
                       ui.tags.td('0'),
                       ui.tags.td('1')
                       ),
                     ui.tags.tr(
                       ui.tags.td('Item2'),
                       ui.tags.td('1'),
                       ui.tags.td('0')
                       ),
                     ui.tags.tr(
                       ui.tags.td('Item3'),
                       ui.tags.td('1'),
                       ui.tags.td('1')
                       )
                     ),
                   class_= 'table table-striped'),  
                 ),               
                 ui.page_fluid(
                   ui.h5('Important Information for Practitioners'),
                      ui.markdown("""
                      - Can currently only support DINO and DINA models (see [the paper here on small sample DCMs](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.621251/full))
                          - DINO = Respondents have at least one skill to solve the question
                          - DINA = Respondents need all the skills to solve the question
                          - Differences between models is apparent when questions measure more than one skill
                      - You'll need a Q-matrix to conduct analyses
                          - A Q-matrix is a checklist that you will create to show what questions (row) measure each skill (column) 
                          - When a question measures one or more skills, you will mark that cell with a 1, otherwise keep it as a 0
                          - **For a Q-matrix template, load your data, choose the number of skills, and click on the generate template button**
                      - Currently only supporting full data datasets
                          - Data must be coded as 1 = Correct and 0 = Incorrect
                          - Missing data could be coded as 0
                      - Currently skills will need to be named A1 - A5
                      - If you have a large sample (N > 200), beta(1,1) priors will be sufficient
                          - Otherwise more informative priors are needed
                          - You can plot what the priors look like by using the `Update parameters` option after choosing values for the prior
                      - Any feedback can be provided as a GitHub issue [here](https://github.com/jpedroza1228/dcm_app/issues)
                      """)
                      ),
                 col_widths = (5, 7)
                 )
               ),
  ui.nav_panel('Data Exploration',
               ui.layout_columns(
                 ui.page_fluid(
                   ui.output_plot('att1_dist'),
                   ui.hr(),
                   ui.output_plot('slip_dist')
                   ),
                 ui.page_fluid(
                   ui.h6('Top 5 rows of dataset'),
                   ui.output_data_frame('dataset'),
                   ui.hr(),
                   ui.h6('Uploaded Q-Matrix'),
                   ui.output_data_frame('qmatrix')
                   ),
                 col_widths = (6, 6)
                 )
               ), 
  ui.nav_spacer(),
  ui.nav_control(ui.input_dark_mode()),
  # ui.nav_panel('C', 'Page C content'),  
  title = '"Small" Sample DCMs For Practitioners',  
  id = 'page',
  sidebar = ui.sidebar(
    ui.input_slider('attr_num', 'Number of skills in your assessment', 2, 5, 2),
    ui.download_button('download_q', 'Generate Q-Matrix Template (Needs data loaded)'),
    ui.input_file('load', 'Load in your data'),
    ui.input_file('qload', 'Load in your Q-Matrix'),
    ui.input_select('type_model',
                    'Choose a Model Type:',
                    {'dino': 'DINO',
                     'dina': 'DINA'}
                    # 'lcdm': 'LCDM'},
    ),
    ui.input_slider('att1_alpha', 'Beta Distribution - Skill 1: Alpha', 0, 50, 20, step = .5),
    ui.input_slider('att1_beta', 'Beta Distribution - Skill 1: Beta', 0, 50, 5, step = .5),
    ui.input_checkbox("all_same_prior", "Keep all the same skill priors as above", True),
    ui.panel_conditional(
      '!input.all_same_prior',
      ui.input_slider('att2_alpha', 'Beta Distribution - Skill 2: Alpha', .5, 50, 20, step = .5),
      ui.input_slider('att2_beta', 'Beta Distribution - Skill 2: Beta', .5, 50, 5, step = .5),
      ui.input_slider('att3_alpha', 'Beta Distribution - Skill 3: Alpha', .5, 50, 20, step = .5),
      ui.input_slider('att3_beta', 'Beta Distribution - Skill 3: Beta', .5, 50, 5, step = .5),
      ui.input_slider('att4_alpha', 'Beta Distribution - Skill 4: Alpha', .5, 50, 20, step = .5),
      ui.input_slider('att4_beta', 'Beta Distribution - Skill 4: Beta', .5, 50, 5, step = .5),
      ui.input_slider('att5_alpha', 'Beta Distribution - Skill 5: Alpha', .5, 50, 20, step = .5),
      ui.input_slider('att5_beta', 'Beta Distribution - Skill 5: Beta', .5, 50, 5, step = .5),
    ),
    ui.hr(),
    ui.h6('How likely students are to have the skill, but get question incorrect (slip)'),
    ui.input_slider('slip_alpha', 'Beta Distribution - Slip: Alpha', .5, 50, 5, step = .5),
    ui.input_slider('slip_beta', 'Beta Distribution - Slip: Beta', .5, 50, 20, step = .5),
    ui.hr(),
    ui.h6('How likely students are to not have the skill, but get question correct (guess)'),
    ui.input_slider('guess_alpha', 'Beta Distribution - Guess: Alpha', .5, 50, 5, step = .5),
    ui.input_slider('guess_beta', 'Beta Distribution - Guess: Beta', .5, 50, 20, step = .5),
    ui.hr(),
    ui.input_action_button('build_model', 'Plot/Update parameters'),
    ui.input_action_button('run_model', 'Run model'),
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
    return df[cols].clean_names(case_type = 'snake')

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
      q = pd.DataFrame(0, index = df.columns, columns = attr_cols).reset_index()
      q.rename(columns={"index": "Item"}, inplace = True)
      empty_q.set(q)
    
  @reactive.calc
  def loaded_q():
    uploaded = input.qload()
    if not uploaded:
        return None
    q = pd.read_csv(uploaded[0]['datapath'])
    q = q.clean_names(case_type = 'snake').rename(columns = {'unnamed_0': 'item'})
    q['item'] = q['item'] + 1
    return q
    
  @render.data_frame
  def qmatrix():
    q = loaded_q()
    if q is None:
        return None
    return render.DataTable(q)
  
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
      + pn.geom_density(color = 'black',
                        fill = 'seagreen')
      + pn.scale_x_continuous(limits = [0, 1],
                              breaks = np.arange(0, 1.1, .1))
      + pn.labs(title = 'How likely is it that students have the skill',
                x = 'Probability',
                y = '',
                caption = 'Plot shows skill 1 (will be similar to other skills)')
      + pn.theme_light()
    )
    return plot.draw()
  
  @render.plot
  @reactive.event(input.build_model)
  def slip_dist():
    alpha = input.slip_alpha()
    beta = input.slip_beta()
    
    if alpha <= 0 or beta <= 0:
        return None
    
    dist_data = pd.DataFrame({
        'value': np.random.beta(alpha, beta, size = 1000)
    })
    
    plot = (
      pn.ggplot(dist_data,
                pn.aes('value'))
      + pn.geom_density(color = 'black',
                        fill = 'seagreen')
      + pn.scale_x_continuous(limits = [0, 1],
                              breaks = np.arange(0, 1.1, .1))
      + pn.labs(title = 'How likely will a student slip',
                x = 'Probability',
                y = '',
                caption = 'Plot shows slip as example (similar to guess)')
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
  
  @reactive.event(input.build_model)
  def stan_parameters_block():
    attr_num = input.attr_num()
    
    param = 'parameters {\nordered[C] raw_nu_ordered;\nvector<lower=0, upper=1>[I] slip;\nvector<lower=0, upper=1>[I] guess;\n'
        
    if attr_num == '2':
      attr_list = ['1', '2']
    elif attr_num == '3':
      attr_list = ['1', '2', '3']
    elif attr_num == '4':
      attr_list = ['1', '2', '3', '4']
    else:
      attr_list = ['1', '2', '3', '4', '5']

    lambdas = '\n'.join([f'real<lower=0, upper=1> lambda{i};' for i in attr_list])

    return param + lambdas + '\n}'
  
  @reactive.event(input.build_model)
  def stan_tparamchunk():
    attr_num = input.attr_num()
    model_type = input.model_type()
    
    param = 'transformed parameters {\nsimplex[C] nu;\nmatrix[I,C] delta;\nmatrix[I,C] pi;'

    if attr_num == '2':
      attr_list = ['1', '2']
      attr_num = [1, 2]
    elif attr_num == '3':
      attr_list = ['1', '2', '3']
      attr_num = [1, 2, 3]
    elif attr_num == '4':
      attr_list = ['1', '2', '3', '4']
      attr_num = [1, 2, 3, 4]
    else:
      attr_list = ['1', '2', '3', '4', '5']
      attr_num = [1, 2, 3, 4, 5]

    thetas = '\n'.join([f'vector[C] theta{i};' for i in attr_list])
    theta_loop_open = '\n\nfor (c in 1:C){\n'
    theta_cal = '\n'.join([f'  theta{i}[c] = (alpha[c, {j}] > 0) ? lambda{i} : (1 - lambda{i});' for i, j in zip(attr_list, attr_num)])
    theta_loop_close = '\n}'

    nu_calc = f'\n\nnu = softmax(raw_nu_ordered);\nvector[C] log_nu = log(nu);\n'

    if model_type == 'dino':
      if attr_list == '2':
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n  pow(1 - theta2[c], Q[i, 2]));\n}}\n}}'
      elif attr_list == '3':
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n  pow(1 - theta2[c], Q[i, 2]) *\n  pow(1 - theta3[c], Q[i, 3]));\n}}\n}}'
      elif attr_list == '4':
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n  pow(1 - theta2[c], Q[i, 2]) *\n  pow(1 - theta3[c], Q[i, 3]) *\n  pow(1 - theta4[c], Q[i, 4]));\n}}\n}}'
      else:
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n  pow(1 - theta2[c], Q[i, 2]) *\n  pow(1 - theta3[c], Q[i, 3]) *\n  pow(1 - theta4[c], Q[i, 4]) *\n  pow(1 - theta5[c], Q[i, 5]));\n}}\n}}'

    elif model_type == 'dina':
      if attr_list == '2':
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = 1 pow(theta1[c], Q[i, 1]) *\n  pow(theta2[c], Q[i, 2]);\n}}\n}}'
      elif attr_list == '3':
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = pow(theta1[c], Q[i, 1]) *\n  pow(theta2[c], Q[i, 2]) *\n  pow(theta3[c], Q[i, 3]);\n}}\n}}'
      elif attr_list == '4':
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = pow(theta1[c], Q[i, 1]) *\n  pow(theta2[c], Q[i, 2]) *\n  pow(theta3[c], Q[i, 3]) *\n  pow(theta4[c], Q[i, 4]);\n}}\n}}'
      else:
        delta_calc = f'for (c in 1:C){{\n  for (i in 1:I){{delta[i, c] = pow(theta1[c], Q[i, 1]) *\n  pow(theta2[c], Q[i, 2]) *\n  pow(theta3[c], Q[i, 3]) *\n  pow(theta4[c], Q[i, 4]) *\n  pow(theta5[c], Q[i, 5]);\n}}\n}}'

    pi_calc = f'for (c in 1:C){{\n  for (i in 1:I){{pi[i,c] = pow((1 - slip[i]), delta[i,c]) *\n  pow(guess[i], (1 - delta[i,c]));\n}}\n}}'

    trans_param = param + thetas + theta_loop_open + theta_cal + theta_loop_close + nu_calc + delta_calc + pi_calc

    return trans_param

  # @render.text
  # def compile_stan():
    
  @render.text
  @reactive.event(input.run_model)
  def model_button():
    return f'{input.run_model()}'
  
  @render.download(filename = 'q_matrix_template.csv')
  def download_q():
    q = empty_q.get()
    # if q is None or q.empty:
    #   raise SafeException('Q-matrix is empty. Load data and edit the table first.')
    yield q.to_csv(index = False)
    
# --------------------------------------------------------------------------------------------------------

app = App(app_ui, server)

# --------------------------------------------------------------------------------------------------------

# extra

