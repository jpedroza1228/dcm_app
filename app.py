import pandas as pd
import matplotlib
import numpy as np
import plotnine as pn
# from great_tables import GT
from janitor import clean_names
import matplotlib.pyplot as plt
from pyhere import here
import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import SafeException
import shinyswatch
from pathlib import Path
import tempfile
from cmdstanpy import CmdStanModel
import joblib

pd.set_option('display.max_columns', None)
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

# --------------------------------------------------------------------------------------------------------

# code
def stan_datachunk():
  data = 'data {\n  int<lower=1> J;\n  int<lower=1> I;\n  int<lower=1> C;\n  int<lower=1> K;\n  matrix<lower=0,upper=1> [J,I] Y;\n  matrix<lower=0,upper=1> [I,K] Q;\n  matrix<lower=0,upper=1> [C,K] alpha;\n}'
  return data

def stan_generatechunk():
  quant = f'generated quantities {{\n  matrix[J,C] prob_resp_class;\n  matrix[J,K] prob_resp_attr;\n  array[I] real eta;\n  row_vector[C] prob_joint;\n  vector[J] log_lik;\n  array[C] real prob_attr_class;\n  matrix[J,I] y_rep;\n\n  for (j in 1:J){{\n   for (c in 1:C){{\n     for(i in 1:I){{\n       real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);\n       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);\n     }}\n     prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));\n     log_lik[j] = log_sum_exp(prob_joint);\n   }}\n   prob_resp_class[j] = prob_joint/sum(prob_joint);\n  }}\n\n  for (j in 1:J){{\n    for (k in 1:K){{\n      for (c in 1:C){{\n        prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];\n      }}\n      prob_resp_attr[j,k] = sum(prob_attr_class);\n    }}\n  }}\n\n  for (j in 1:J){{\n    int z = categorical_rng(nu);\n    for (i in 1:I){{\n      y_rep[j,i] = bernoulli_rng(pi[i,z]);\n    }}\n  }}\n}}'
  
  return quant

def stan_generateprior():
  quant = f'generated quantities {{\n  matrix[J,I] y_rep;\n\n  for (j in 1:J){{\n    int z = categorical_rng(nu);\n    for (i in 1:I){{\n      y_rep[j,i] = bernoulli_rng(pi[i,z]);\n    }}\n  }}\n}}'
  
  return quant

# def acceptable_fit_stat(inference_data, func_name = ['waic', 'loo']):
#   if func_name == 'waic':
#     est = np.abs(az.waic(inference_data).iloc[0])
#     se = az.waic(inference_data).iloc[1]
    
#     if est > se * 2.5:
#       print('Absolute difference is greater than 2.5 x the standard error of the difference. Model is acceptable.')
      
#     else:
#       print('Absolute difference is not greater than 2.5 x the standard error of the difference. Model is not acceptable.')
#   elif func_name == 'loo':
#     est = np.abs(az.loo(inference_data).iloc[0])
#     se = az.loo(inference_data).iloc[1]
    
#     if est > se * 2.5:
#       print('Absolute difference is greater than 2.5 x the standard error of the difference. Model is acceptable.')
      
#     else:
#       print('Absolute difference is not greater than 2.5 x the standard error of the difference. Model is not acceptable.')

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
                      - If you have a sample greater than 200, beta(20,5) priors will be sufficient for skills
                          - You can plot what the priors look like by using the `plot priors` option after choosing values for the prior
                      - Any feedback can be provided as a GitHub issue [here](https://github.com/jpedroza1228/dcm_app/issues)
                      """)
                      ),
                 col_widths = (5, 7)
                 )
               ),
    ui.nav_panel('Exploring Priors',
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
    ui.nav_panel('Quick Model Diagnostics',
               ui.page_fluid(
                 ui.output_table('top_rhat_values')
               )
               ),
  ui.nav_spacer(),
  ui.nav_control(ui.input_dark_mode()),
  sidebar = ui.sidebar(
    ui.input_checkbox('have_df', 'I have a dataframe with 1s (correct) and 0s (incorrect)', True),
    ui.panel_conditional(
      '!input.have_df',
      ui.input_file('messy_load', 'Load in your raw data'),
      ui.input_text('correct_answers', 'Input correct responses separated by commas (e.g., a, d, c...)')
    ),
    ui.panel_conditional(
      'input.have_df',
      ui.input_file('load', 'Load in your data'),
    ),
    ui.input_checkbox('have_qmatrix', 'I have a Q-matrix', False),
    ui.panel_conditional(
      '!input.have_qmatrix',
      ui.input_slider('attr_num', 'Number of skills in your assessment', 2, 5, 2),
      ui.download_button('download_q', 'Generate Q-Matrix Template')
    ),
    ui.panel_conditional(
      'input.have_qmatrix',
      ui.input_file('qload', 'Load in your Q-Matrix')
    ),
    ui.hr(),
    ui.input_select('type_model',
                    'Choose a Model Type:',
                    {'dino': 'DINO',
                     'dina': 'DINA'}
    ),
    ui.h6('How likely are students to have the skill?'),
    ui.input_slider('att1_alpha', 'Beta Distribution - Skill 1: Alpha', 0, 50, 20, step = .5),
    ui.input_slider('att1_beta', 'Beta Distribution - Skill 1: Beta', 0, 50, 5, step = .5),
    ui.input_checkbox('all_same_prior', 'Keep all the same skill priors as above', True),
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
    # ui.hr(),
    ui.h6('How likely are students to slip?'),
    ui.input_slider('slip_alpha', 'Beta Distribution - Slip: Alpha', .5, 50, 5, step = .5),
    ui.input_slider('slip_beta', 'Beta Distribution - Slip: Beta', .5, 50, 20, step = .5),
    # ui.hr(),
    ui.h6('How likely are students to guess?'),
    ui.input_slider('guess_alpha', 'Beta Distribution - Guess: Alpha', .5, 50, 5, step = .5),
    ui.input_slider('guess_beta', 'Beta Distribution - Guess: Beta', .5, 50, 20, step = .5),
    # ui.hr(),
    ui.input_action_button('plot_param', 'Plot priors'),
    ui.hr(),
    ui.input_action_button('build_model', 'Update parameters'),
    # ui.input_slider('threshold', 'Probability Threshold for Skill Attainment (Higher means less false positives)', 0, 1, .8, step = .05),
    ui.input_action_button('run_model', 'Run model'),
    ui.input_checkbox('use_init_values', 'Check this box if model does not converge', False)
  ),
  title = '"Small" Sample DCMs',  
  id = 'page',
  theme = shinyswatch.theme.flatly
)

# --------------------------------------------------------------------------------------------------------

def server(input: Inputs, output: Outputs, session: Session):
  empty_q = reactive.Value(pd.DataFrame())
  
  pass

  compiled_model = reactive.Value(None)
  compiled_stan_path = reactive.Value(None)
  model_fit = reactive.Value(None)
  # idata = reactive.Value(None)
  
  compiled_prior = reactive.Value(None)
  compiled_prior_path = reactive.Value(None)
  prior_fit = reactive.Value(None)
  # iprior = reactive.Value(None) # don't think I need this actually since I combined prior and full datasets for idata

  @reactive.calc
  def create_binary_df():
    input_values = input.correct_answers()
    input_list = input_values.strip(',')
    upload_df = input.messy_load()
    
    rawdf = pd.read_csv(upload_df[0]['datapath'])
    rawdf.columns = [f'item{i}' for i in range(1, len(rawdf.columns) + 1)]
    
    binary_df = pd.DataFrame({i: np.where(rawdf[i] == j, 1, 0) for i, j in zip(rawdf.columns, input_list)})
    
    # this needs up to be updated
    return binary_df

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

    df = df.loc[:, cols].copy()
    df.columns = [f'item{i}' for i in range(1, len(df.columns) + 1)]

    return df

  @reactive.effect
  def _initialize_qmatrix():
    df = loaded_data()
    if df is not None:
      n_attrs = input.attr_num()
      attr_cols = [f'A{i}' for i in range(1, n_attrs + 1)]
      # Build the fresh grid
      q = pd.DataFrame(0, index = df.columns, columns = attr_cols).reset_index()
      q.rename(columns = {'index': 'Item'}, inplace = True)
      empty_q.set(q)
      
  @render.download(filename = 'q_matrix_template.csv')
  def download_q():
    q = empty_q.get()
    # if q is None or q.empty:
    #   raise SafeException('Q-matrix is empty. Load data and edit the table first.')
    yield q.to_csv(index = False)
    
  @reactive.calc
  def loaded_q():
    uploaded = input.qload()
    if not uploaded:
        return None
      
    q = pd.read_csv(uploaded[0]['datapath']).clean_names(case_type = 'snake')
    q = q.rename(columns={q.columns[0]: 'item'})
    q['item'] = [f'item{i}' for i in range(1, len(q) + 1)]

    return q
    
  @render.data_frame
  def qmatrix():
    q = loaded_q()
    if q is None:
        return None
    return render.DataTable(q)
  
  @render.data_frame
  def dataset():
    df = loaded_data()
    if df is None:
        return None
    return render.DataTable(df.head(),
                            height = '225px')
    
  @render.text
  def att1_avalue():
    return f'{input.att1_alpha()}'
  
  @render.text
  def att1_bvalue():
    return f'{input.att1_beta()}'
  
  @render.plot
  @reactive.event(input.plot_param)
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
  @reactive.event(input.plot_param)
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
  
  @reactive.calc
  def create_alpha():
    q = loaded_q()
    n = q.shape[1] - 1
    
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
  
  @reactive.calc
  def stan_data_dict():
    df = loaded_data()
    q = loaded_q()
    alpha = create_alpha()
    
    if df is None or q is None or alpha is None:
        return None
    
    stan_dict = {
      'J': df.shape[0],
      'I': df.shape[1],
      'C': alpha.shape[0],
      'K': q.shape[1] - 1,
      'Y': df.to_numpy(),
      'Q': q.iloc[:, 1:].to_numpy(),
      'alpha': alpha.to_numpy()
      }
    
    return stan_dict
  
  @reactive.calc
  @reactive.event(input.build_model)
  def get_inits():
    if not input.use_init_values():
        return None
      
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
      if input.use_init_values() == True:
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
      if input.use_init_values() == True:
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
      if input.use_init_values() == True:
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
      if input.use_init_values() == True:
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
  def stan_paramchunk():
    q = loaded_q()
    attr_num = q.shape[1] - 1
    
    param = 'parameters {\n  ordered[C] raw_nu_ordered;\n  vector<lower=0, upper=1>[I] slip;\n  vector<lower=0, upper=1>[I] guess;\n'
        
    if attr_num == 2:
      attr_list = [1, 2]
    elif attr_num == 3:
      attr_list = [1, 2, 3]
    elif attr_num == 4:
      attr_list = [1, 2, 3, 4]
    else:
      attr_list = [1, 2, 3, 4, 5]

    lambdas = '\n'.join([f'  real<lower=0, upper=1> lambda{i};' for i in attr_list])

    return param + lambdas + '\n}'
  
  @reactive.event(input.build_model)
  def stan_tparamchunk():
    q = loaded_q()
    attr_num = q.shape[1] - 1
    type_model = input.type_model()
    
    param = 'transformed parameters {\n  simplex[C] nu;\n  matrix[I,C] delta;\n  matrix[I,C] pi;\n'

    if attr_num == 2:
      attr_list = [1, 2]
    elif attr_num == 3:
      attr_list = [1, 2, 3]
    elif attr_num == 4:
      attr_list = [1, 2, 3, 4]
    else:
      attr_list = [1, 2, 3, 4, 5]

    thetas = '\n'.join([f'  vector[C] theta{i};' for i in attr_list])
    theta_loop_open = '\n\n  for (c in 1:C){\n'
    theta_calc = '\n'.join([f'    theta{i}[c] = (alpha[c, {i}] > 0) ? lambda{i} : (1 - lambda{i});' for i in attr_list])
    theta_loop_close = '\n  }'

    nu_calc = f'\n\n  nu = softmax(raw_nu_ordered);\n  vector[C] log_nu = log(nu);\n\n'

    if type_model == 'dino':
      if attr_num == 2:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n      pow(1 - theta2[c], Q[i, 2]));\n    }}\n  }}\n'
      elif attr_num == 3:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n      pow(1 - theta2[c], Q[i, 2]) *\n      pow(1 - theta3[c], Q[i, 3]));\n    }}\n  }}\n'
      elif attr_num == 4:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n      pow(1 - theta2[c], Q[i, 2]) *\n  pow(1 - theta3[c], Q[i, 3]) *\n      pow(1 - theta4[c], Q[i, 4]));\n    }}\n  }}\n'
      else:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = 1 - (pow(1 - theta1[c], Q[i, 1]) *\n      pow(1 - theta2[c], Q[i, 2]) *\n      pow(1 - theta3[c], Q[i, 3]) *\n      pow(1 - theta4[c], Q[i, 4]) *\n      pow(1 - theta5[c], Q[i, 5]));\n    }}\n  }}\n'

    elif type_model == 'dina':
      if attr_num == 2:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = 1 pow(theta1[c], Q[i, 1]) *\n      pow(theta2[c], Q[i, 2]);\n    }}\n  }}\n'
      elif attr_num == 3:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = pow(theta1[c], Q[i, 1]) *\n      pow(theta2[c], Q[i, 2]) *\n      pow(theta3[c], Q[i, 3]);\n    }}\n  }}\n'
      elif attr_num == 4:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = pow(theta1[c], Q[i, 1]) *\n      pow(theta2[c], Q[i, 2]) *\n      pow(theta3[c], Q[i, 3]) *\n      pow(theta4[c], Q[i, 4]);\n    }}\n  }}\n'
      else:
        delta_calc = f'  for (c in 1:C){{\n    for (i in 1:I){{\n      delta[i, c] = pow(theta1[c], Q[i, 1]) *\n      pow(theta2[c], Q[i, 2]) *\n      pow(theta3[c], Q[i, 3]) *\n      pow(theta4[c], Q[i, 4]) *\n      pow(theta5[c], Q[i, 5]);\n    }}\n  }}\n'

    pi_calc = f'\n  for (c in 1:C){{\n    for (i in 1:I){{\n      pi[i,c] = pow((1 - slip[i]), delta[i,c]) *\n      pow(guess[i], (1 - delta[i,c]));\n    }}\n  }}\n}}'

    trans_param = param + thetas + theta_loop_open + theta_calc + theta_loop_close + nu_calc + delta_calc + pi_calc

    return trans_param
  
  @reactive.event(input.build_model)
  def stan_modelchunk():
    q = loaded_q()
    n = q.shape[1] - 1
    slip_alpha = input.slip_alpha()
    slip_beta = input.slip_beta()
    guess_alpha = input.guess_alpha()
    guess_beta = input.guess_beta()
  
    model_start = f'model{{\n  array[C] real ps;\n  array[I] real eta;\n\n  raw_nu_ordered ~ normal(0,2);\n  for (i in 1:I){{\n    slip[i] ~ beta({slip_alpha}, {slip_beta});\n    guess[i] ~ beta({guess_alpha}, {guess_beta});\n  }}\n'

    if n == 2:
      alpha1 = input.att1_alpha()
      beta1 = input.att1_beta()
      alpha2 = input.att2_alpha()
      beta2 = input.att2_beta()

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n\n'

    elif n == 3:
      alpha1 = input.att1_alpha()
      beta1 = input.att1_beta()
      alpha2 = input.att2_alpha()
      beta2 = input.att2_beta()
      alpha3 = input.att3_alpha()
      beta3 = input.att3_beta()

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n  lambda3 ~ beta({alpha3}, {beta3});\n\n'

    elif n == 4:
      alpha1 = input.att1_alpha()
      beta1 = input.att1_beta()
      alpha2 = input.att2_alpha()
      beta2 = input.att2_beta()
      alpha3 = input.att3_alpha()
      beta3 = input.att3_beta()
      alpha4 = input.att4_alpha()
      beta4 = input.att4_beta()

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n  lambda3 ~ beta({alpha3}, {beta3});\n  lambda4 ~ beta({alpha4}, {beta4});\n\n'

    elif n == 5:
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

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n  lambda3 ~ beta({alpha3}, {beta3});\n  lambda4 ~ beta({alpha4}, {beta4});\n  lambda5 ~ beta({alpha5}, {beta5});\n\n'

    model_end = f'  for (j in 1:J){{\n    for (c in 1:C){{\n      for (i in 1:I){{\n        real p = fmin(fmax(pi[i,c], 1e-9), 1 - 1e-9);\n        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);\n      }}\n      ps[c] = log_nu[c] + sum(eta);\n    }}\n    target += log_sum_exp(ps);\n  }}\n}}'
  
    model = model_start + priors + model_end

    return model
  
  @reactive.event(input.build_model)
  def stan_priormodelchunk():
    q = loaded_q()
    n = q.shape[1] - 1
    slip_alpha = input.slip_alpha()
    slip_beta = input.slip_beta()
    guess_alpha = input.guess_alpha()
    guess_beta = input.guess_beta()
  
    model_start = f'model{{\n  array[C] real ps;\n  array[I] real eta;\n\n  raw_nu_ordered ~ normal(0,2);\n  for (i in 1:I){{\n    slip[i] ~ beta({slip_alpha}, {slip_beta});\n    guess[i] ~ beta({guess_alpha}, {guess_beta});\n  }}\n'

    if n == 2:
      alpha1 = input.att1_alpha()
      beta1 = input.att1_beta()
      alpha2 = input.att2_alpha()
      beta2 = input.att2_beta()

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n\n'

    elif n == 3:
      alpha1 = input.att1_alpha()
      beta1 = input.att1_beta()
      alpha2 = input.att2_alpha()
      beta2 = input.att2_beta()
      alpha3 = input.att3_alpha()
      beta3 = input.att3_beta()

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n  lambda3 ~ beta({alpha3}, {beta3});\n\n'

    elif n == 4:
      alpha1 = input.att1_alpha()
      beta1 = input.att1_beta()
      alpha2 = input.att2_alpha()
      beta2 = input.att2_beta()
      alpha3 = input.att3_alpha()
      beta3 = input.att3_beta()
      alpha4 = input.att4_alpha()
      beta4 = input.att4_beta()

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n  lambda3 ~ beta({alpha3}, {beta3});\n  lambda4 ~ beta({alpha4}, {beta4});\n\n'

    elif n == 5:
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

      priors = f'  lambda1 ~ beta({alpha1}, {beta1});\n  lambda2 ~ beta({alpha2}, {beta2});\n  lambda3 ~ beta({alpha3}, {beta3});\n  lambda4 ~ beta({alpha4}, {beta4});\n  lambda5 ~ beta({alpha5}, {beta5});\n\n'
  
    model = model_start + priors + '}'

    return model

  @reactive.effect
  @reactive.event(input.build_model)
  def build_stan_program():
    chunks = [
        stan_datachunk(),
        stan_paramchunk(),
        stan_tparamchunk(),
        stan_modelchunk(),
        stan_generatechunk()
    ]
    
    stan_code = '\n\n'.join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
    
    if not stan_code:
      return
    
    stan_dir = Path(here('stan_files'))
    stan_dir.mkdir(parents = True,
                   exist_ok = True)
        
    file_name = 'model.stan'
    stan_file = stan_dir / file_name
    stan_file.write_text(stan_code, encoding = 'utf-8')
        
    model = CmdStanModel(
      stan_file = str(stan_file),
      cpp_options = {'STAN_THREADS': 'TRUE'}
    )

    compiled_model.set(model)
    compiled_stan_path.set(stan_file)
    
  @reactive.effect
  @reactive.event(input.build_model)
  def build_stan_prioronly():
    chunks = [
        stan_datachunk(),
        stan_paramchunk(),
        stan_tparamchunk(),
        stan_priormodelchunk(),
        stan_generateprior()
    ]
    
    stan_code = '\n\n'.join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
    
    if not stan_code:
      return
    
    stan_dir = Path(here('stan_files'))
    stan_dir.mkdir(parents = True,
                   exist_ok = True)
        
    file_name = 'prior_model.stan'
    stan_file = stan_dir / file_name
    stan_file.write_text(stan_code, encoding = 'utf-8')
        
    model = CmdStanModel(
      stan_file = str(stan_file),
      cpp_options = {'STAN_THREADS': 'TRUE'}
    )

    compiled_prior.set(model)
    compiled_prior_path.set(stan_file)
    
  @reactive.effect
  @reactive.event(input.run_model)
  def run_model():
    model = compiled_model.get()
    prior_model = compiled_prior.get()
    
    if model is None or prior_model is None:
      raise SafeException('Compile the Stan model first.')
    
    with ui.Progress(min = 1, max = 20) as p:
        p.set(message = 'MCMC Sampling in progress',
              detail = 'Model is running')
    
        if input.use_init_values() == True:
          fit = model.sample(
            data = stan_data_dict(),
            inits = get_inits(),
            adapt_delta = .99,
            chains = 4,
            parallel_chains = 4,
            iter_warmup = 2000,
            iter_sampling = 2000
          )
          model_fit.set(fit)
          
          p.set(15,
                detail = 'Running prior-only model.')
          
          pfit = model.sample(
            data = stan_data_dict(),
            inits = get_inits(),
            adapt_delta = .99,
            chains = 4,
            parallel_chains = 4,
            iter_warmup = 2000,
            iter_sampling = 2000
          )
          prior_fit.set(pfit)
          
          p.set(20,
                detail = 'Completed both models.')

        else:
          fit = model.sample(
            data = stan_data_dict(),
            chains = 4,
            parallel_chains = 4,
            iter_warmup = 2000,
            iter_sampling = 2000
          )
          model_fit.set(fit)
          
          p.set(15,
                detail = 'Running prior-only model.')
          
          pfit = model.sample(
            data = stan_data_dict(),
            chains = 4,
            parallel_chains = 4,
            iter_warmup = 2000,
            iter_sampling = 2000
          )
          prior_fit.set(pfit)
          
          p.set(20,
                detail = 'Completed both models.')

  @reactive.calc
  def diagnostic_summary():
    fit = model_fit.get()
    pfit = prior_fit.get()
    
    if fit is None:
      return None
    
    if pfit is None:
      return None
    
    diag_dir = Path(here('diagnostics'))
    diag_dir.mkdir(parents = True,
                   exist_ok = True)
    
    summary_df = pd.DataFrame(fit.summary())
    prior_df = pd.DataFrame(pfit.summary())
    
    summary_df.to_csv(here(f'{diag_dir}/model_diagnostics.csv'))
    prior_df.to_csv(here(f'{diag_dir}/prior_diagnostics.csv'))
    
    return summary_df

  @reactive.effect
  @reactive.event(input.run_model)
  def save_data_models_fits():
    df = loaded_data()
    q = loaded_q()
    n = q.shape[1] - 1
    alpha = create_alpha()
    
    model = compiled_model.get()
    prior_model = compiled_prior.get()
    fit = model_fit.get()
    pfit = prior_fit.get()
    
    job_dir = Path(here('data_model_fit'))
    job_dir.mkdir(parents = True,
                   exist_ok = True)
    
    df.to_csv(f'{job_dir}/df.csv')
    q.to_csv(f'{job_dir}/q.csv')
    alpha.to_csv(f'{job_dir}/alpha.csv')
    
    (joblib.dump([model, fit],
                 here(f'{job_dir}/modfit.joblib'),
                 compress = 3))
    
    (joblib.dump([prior_model, pfit],
                 here(f'{job_dir}/modfit_prior.joblib'),
                 compress = 3))
    
  
  @render.table
  def top_rhat_values():
    df = diagnostic_summary()
    
    if df is None or 'R_hat' not in df.columns:
        return pd.DataFrame()
    
    return df.sort_values('R_hat',
                          ascending = False).head(5).reset_index()

# save diagnostics

# save model and fit

  # @reactive.calc
  # def create_idata():
  #   fit = model_fit.get()
  #   pfit = prior_fit.get()
  #   df = loaded_data()
  #   df = df.filter(regex = 'item')
    
  #   idcm = azb.from_cmdstanpy(
  #     posterior = fit,
  #     prior = pfit,
  #     posterior_predictive = ['y_rep'],
  #     prior_predictive = ['y_rep'],
  #     observed_data = {'y_rep': df},
  #     log_likelihood = {'Y': 'eta'}
  #     )
    
  #   # idata.set(idcm)   # set the reactive.Value correctly
  #   return idcm 
  
  # @render.image
  # def guess_prior_post_plot():
  #   # idcm = idata.get()
  #   idcm = create_idata()
    
  #   guess_plot = azp.plot_prior_posterior(idcm,
  #                                         var_names = ['guess'],
  #                                         kind = 'kde',
  #                                         backend = 'matplotlib')
    
  #   azp_dir = Path(here('arviz_plots'))
  #   azp_dir.mkdir(parents = True,
  #                  exist_ok = True)
        
  #   file_name = 'guess_plots.png'
  #   plot_file = azp_dir / file_name
    
  #   fig = plt.gcf()
  #   fig.savefig(plot_file, format = 'png', bbox_inches = 'tight')
  #   plt.close(fig)  # Crucial for memory management
        
  #   return {'src': str(plot_file), 'width': '600px'}
  
  # @render.image
  # def slip_prior_post_plot():
  #   # idcm = idata.get()
  #   idcm = create_idata()
    
  #   slip_plot = azp.plot_prior_posterior(idcm,
  #                                         var_names = ['slip'],
  #                                         kind = 'kde')
    
  #   azp_dir = Path(here('arviz_plots'))
  #   azp_dir.mkdir(parents = True,
  #                  exist_ok = True)
        
  #   file_name = 'slip_plots.png'
  #   plot_file = azp_dir / file_name
    
  #   fig = plt.gcf()
  #   fig.savefig(plot_file, format = 'png', bbox_inches = 'tight')
  #   plt.close(fig)  # Crucial for memory management
        
  #   return {'src': str(plot_file), 'width': '600px'}
  
  # @render.image
  # def nu_prior_post_plot():
  #   # idcm = idata.get()
  #   idcm = create_idata()
    
  #   nu_plot = azp.plot_prior_posterior(idcm,
  #                                         var_names = ['nu'],
  #                                         kind = 'kde')
    
  #   azp_dir = Path(here('arviz_plots'))
  #   azp_dir.mkdir(parents = True,
  #                  exist_ok = True)
        
  #   file_name = 'nu_plots.png'
  #   plot_file = azp_dir / file_name
    
  #   fig = plt.gcf()
  #   fig.savefig(plot_file, format = 'png', bbox_inches = 'tight')
  #   plt.close(fig)  # Crucial for memory management
        
  #   return {'src': str(plot_file), 'width': '600px'}
  
  # @render.image
  # def lambda_prior_post_plot():
  #   # idcm = idata.get()
  #   idcm = create_idata()
  #   n = input.attr_num()
    
  #   if n == 2:
  #     lambda_plot = azp.plot_prior_posterior(idcm,
  #                                            var_names = ['lambda1', 'lambda2'],
  #                                            kind = 'kde')
  #   elif n == 3:
  #     lambda_plot = azp.plot_prior_posterior(idcm,
  #                                            var_names = ['lambda1', 'lambda2', 'lambda3'],
  #                                            kind = 'kde')
  #   elif n == 4:
  #     lambda_plot = azp.plot_prior_posterior(idcm,
  #                                            var_names = ['lambda1', 'lambda2', 'lambda3', 'lambda4'],
  #                                            kind = 'kde')
  #   elif n == 5:
  #     lambda_plot = azp.plot_prior_posterior(idcm,
  #                                            var_names = ['lambda1', 'lambda2', 'lambda3', 'lambda4', 'lambda5'],
  #                                            kind = 'kde')
      
  #   azp_dir = Path(here('arviz_plots'))
  #   azp_dir.mkdir(parents = True,
  #                  exist_ok = True)
        
  #   file_name = 'lambda_plots.png'
  #   plot_file = azp_dir / file_name
    
  #   fig = plt.gcf()
  #   fig.savefig(plot_file, format = 'png', bbox_inches = 'tight')
  #   plt.close(fig)  # Crucial for memory management
        
  #   return {'src': str(plot_file), 'width': '600px'}
    
    
# --------------------------------------------------------------------------------------------------------

app = App(app_ui, server)

# --------------------------------------------------------------------------------------------------------

# 