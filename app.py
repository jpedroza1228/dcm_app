import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotnine as pn
import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import SafeException
import shinyswatch
from pathlib import Path
import tempfile
import subprocess
import os
import sys
from cmdstanpy import CmdStanModel
import joblib
from janitor import clean_names
from pyhere import here

pd.set_option('display.max_columns', None)
matplotlib.rcParams.update({'savefig.bbox': 'tight'})

# ---------------------------------------------------------------------------------------------------

app_ui = ui.page_navbar(
  ui.nav_panel('Background Information',
               ui.layout_columns(
                 ui.page_fluid(
                   ui.markdown("""
                               This page has most of the information required to use the following pages for using diagnostic classification models (DCM) to see which respondents have the skills measured in assessments. Currently, this application is focused on DCMs for smaller samples as an introduction to using DCMs as an preventive measure to flag respondents that may not have a grasp of skills assessed in an assessment.
                             
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
                      - Model structure types include: 
                          - unconstrained: allow the skills to be related
                          - linear: there is a belief that skill 1 is needed for skill 2 and so on (e.g., Skill1 --> Skill2 --> Skill3...)
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
                   ui.panel_conditional(
                     "input.prof_model == 'linear'",
                     ui.output_plot('att1_dist'),
                   ),
                    ui.hr(),
                    ui.output_plot('slip_dist'),
                    ui.hr(),
                    ui.output_plot('guess_dist')
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
    ui.input_select('prof_model',
                    'Choose relationship between skills',
                    {'unconstrained': 'Unconstrained',
                     'linear': 'Linear'}),
    ui.input_select('type_model',
                    'Choose a Model Type:',
                    {'dino': 'DINO',
                     'dina': 'DINA'}
    ),
    ui.panel_conditional(
      "input.prof_model == 'linear'",
      ui.h6('How likely is a respondent to have the skill?'),
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
    )
    ),
    ui.h6('How likely is a respondent to slip?'),
    ui.input_slider('slip_alpha', 'Beta Distribution - Slip: Alpha', .5, 50, 5, step = .5),
    ui.input_slider('slip_beta', 'Beta Distribution - Slip: Beta', .5, 50, 20, step = .5),
    ui.h6('How likely is a respondent to guess?'),
    ui.input_slider('guess_alpha', 'Beta Distribution - Guess: Alpha', .5, 50, 5, step = .5),
    ui.input_slider('guess_beta', 'Beta Distribution - Guess: Beta', .5, 50, 20, step = .5),
    ui.input_action_button('plot_param', 'Plot priors'),
    ui.hr(),
    ui.input_action_button('build_model', 'Update parameters (after choosing priors)'),
    ui.input_action_button('run_model', 'Run model'),
    ui.input_checkbox('use_init_values', 'Check this box if model does not converge', False),
    ui.input_slider('threshold',
                    'Probability threshold to decide is respondents have skill', 0.5, 0.95, 0.8, step = .01),
    ui.download_button('download_report', 'Download Report')
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
      attr_cols = [f'attr{i}' for i in range(1, n_attrs + 1)]
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
    q = q.rename(columns = {q.columns[0]: 'item'})
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
      + pn.labs(title = 'How likely is it that a respondent will have the skill',
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
      + pn.labs(title = 'How likely is it that a respondent will slip',
                x = 'Probability',
                y = '')
      + pn.theme_light()
    )
    return plot.draw()
  
  @render.plot
  @reactive.event(input.plot_param)
  def guess_dist():
    alpha = input.guess_alpha()
    beta = input.guess_beta()
    
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
      + pn.labs(title = 'How likely is it that a respondent will guess',
                x = 'Probability',
                y = '')
      + pn.theme_light()
    )
    return plot.draw()
  
  @reactive.calc
  def create_alpha():
    q = loaded_q()
    # n = q.shape[1] - 1
    n = q.filter(regex = 'attr').shape[1]
    prof_model = input.prof_model()
    
    if n == 2:
      alpha = pd.DataFrame([(a, b) for a in np.arange(2) for b in np.arange(2)])
      alpha = alpha.rename(columns = {0: 'attr1',
                                      1: 'attr2'})
      
      if prof_model == 'linear':
        alpha = alpha.loc[~((alpha['attr1'] == 0) & (alpha['attr2'] == 1))]
        alpha = alpha.reset_index(drop = True)
      
    elif n == 3:
      alpha = pd.DataFrame([(a, b, c) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'attr1',
                                      1: 'attr2',
                                      2: 'attr3'})
      
      if prof_model == 'linear':
        alpha = (
          alpha.loc[~((alpha['attr1'] == 0) 
                     & (alpha['attr2'] == 1)) 
                   & ~((alpha['attr2'] == 0) 
                       & (alpha['attr3'] == 1))]
        )
        alpha = alpha.reset_index(drop = True)
              
    elif n == 4:
      alpha = pd.DataFrame([(a, b, c, d) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'attr1',
                                      1: 'attr2',
                                      2: 'attr3',
                                      3: 'attr4'})
      
      if prof_model == 'linear':
        alpha = (
          alpha.loc[~((alpha['attr1'] == 0) 
                     & (alpha['attr2'] == 1)) 
                   & ~((alpha['attr2'] == 0) 
                       & (alpha['attr3'] == 1)) 
                   & ~((alpha['attr3'] == 0) 
                       & (alpha['attr4'] == 1))]
        )
        alpha = alpha.reset_index(drop = True)
    
    elif n == 5:
      alpha = pd.DataFrame([(a, b, c, d, e) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2) for e in np.arange(2)])
      
      alpha = alpha.rename(columns = {0: 'attr1',
                                      1: 'attr2',
                                      2: 'attr3',
                                      3: 'attr4',
                                      4: 'attr5'})
      
      if prof_model == 'linear':
        alpha = (
          alpha.loc[~((alpha['attr1'] == 0) 
                     & (alpha['attr2'] == 1)) 
                   & ~((alpha['attr2'] == 0) 
                       & (alpha['attr3'] == 1)) 
                   & ~((alpha['attr3'] == 0) 
                       & (alpha['attr4'] == 1))
                   & ~((alpha['attr4'] == 0) 
                       & (alpha['attr5'] == 1))]
        )
        alpha = alpha.reset_index(drop = True)
        
    return alpha
  
  @reactive.calc
  def create_xi():
    type_model = input.type_model()
    q = loaded_q()
    alpha = create_alpha()
    
    xi = np.zeros((q.shape[0], alpha.shape[0]), dtype = int)
    # q = q.iloc[:, 1:]
    q = q.filter(regex = 'attr')
    q_arr = q.to_numpy()
    alpha_arr = alpha.to_numpy()
    
    if type_model == 'dino':
      for i in range(q_arr.shape[0]):
        for c in range(alpha_arr.shape[0]):
          if np.any((q_arr[i, :] == 1) & (alpha_arr[c, :] == 1)):
            xi[i, c] = 1
            
    elif type_model == 'dina':
      for i in range(q_arr.shape[0]):
        # Get the indices where the item requires an attribute
        req_attr = np.where(q_arr[i, :] == 1)[0]

        for c in range(alpha_arr.shape[0]):
          # Check if the class has a 1 at ALL those required indices
          if np.all(alpha_arr[c, req_attr] == 1):
              xi[i, c] = 1
              
    else:
        raise ValueError("Structural model must be 'DINA' or 'DINO'")
  
    return xi

  @reactive.calc
  def stan_data_dict():
    df = loaded_data()
    q = loaded_q()
    alpha = create_alpha()
    xi = create_xi()
    
    if df is None or q is None or alpha is None or xi is None:
      return None
    
    stan_dict = {
      'J': df.shape[0],
      'I': df.shape[1],
      'C': alpha.shape[0],
      # 'K': q.shape[1] - 1,
      'K': q.filter(regex = 'attr').shape[1],
      'Y': df.to_numpy(),
      # 'Q': q.iloc[:, 1:].to_numpy(),
      'Q': q.filter(regex = 'attr').to_numpy(),
      'alpha': alpha.to_numpy(),
      'xi': xi
      }
    
    return stan_dict
  
  @reactive.calc
  @reactive.event(input.build_model)
  def get_inits():
    if not input.use_init_values():
        return None
      
    q = loaded_q()
    n = q.filter(regex = 'attr').shape[1]
    df = loaded_data()
    alpha = create_alpha()
    prof_model = input.prof_model()
    init_values = input.use_init_values()
    
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
    
    if prof_model == 'linear':
      if n == 2:
        if init_values == True:
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
        if init_values == True:
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
        if init_values == True:
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
        if init_values == True:
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
        
      elif prof_model == 'unconstrained':
        return {
              'nu': np.repeat(1/alpha.shape[0], alpha.shape[0]),
              'slip': np.clip(np.random.uniform((slip_start - slip_sd), (slip_start + slip_sd), size = df.shape[1]).tolist(), 0, 1),
              'guess': np.clip(np.random.uniform((guess_start - guess_sd), (guess_start + guess_sd), size = df.shape[1]).tolist(), 0, 1)
              }
    
  @reactive.event(input.build_model)
  def update_model():
    return f'{input.build_model()}'
  
  @reactive.effect
  @reactive.event(input.build_model)
  def show_build_notification():
    ui.notification_show('Parameters are updated. Model is ready to run.',
                         type = 'message',
                         duration = 10)
  
  @reactive.event(input.build_model)
  def stan_datachunk():
    data = 'data {\n  int<lower=1> J;\n  int<lower=1> I;\n  int<lower=1> C;\n  int<lower=1> K;\n  matrix<lower=0,upper=1> [J,I] Y;\n  matrix<lower=0,upper=1> [I,K] Q;\n  matrix<lower=0,upper=1> [C,K] alpha;\n  matrix<lower=0,upper=1> [I,C] xi;\n}'
    return data
  
  @reactive.event(input.build_model)
  def stan_generatechunk():
    prof_model = input.prof_model()

    if prof_model == 'linear':
      quant = f'generated quantities {{\n  matrix[J,C] prob_resp_class;\n  matrix[J,K] prob_resp_attr;\n  array[I] real eta;\n  row_vector[C] prob_joint;\n  vector[J] log_lik;\n  array[C] real prob_attr_class;\n  matrix[J,I] y_rep;\n\n  for (j in 1:J){{\n   for (c in 1:C){{\n     for(i in 1:I){{\n       real p = fmin(fmax(pi[i,c], 1e-9), (1 - 1e-9));\n       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);\n     }}\n     prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));\n     log_lik[j] = log_sum_exp(prob_joint);\n   }}\n   prob_resp_class[j] = prob_joint/sum(prob_joint);\n  }}\n\n  for (j in 1:J){{\n    for (k in 1:K){{\n      for (c in 1:C){{\n        prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];\n      }}\n      prob_resp_attr[j,k] = sum(prob_attr_class);\n    }}\n  }}\n\n  for (j in 1:J){{\n    int z = categorical_rng(nu/sum(nu));\n    for (i in 1:I){{\n      y_rep[j,i] = bernoulli_rng(pi[i,z]);\n    }}\n  }}\n}}'

    elif prof_model == 'unconstrained':
      quant = f'generated quantities {{\n  matrix[J,C] prob_resp_class;\n  matrix[J,K] prob_resp_attr;\n  array[I] real eta;\n  row_vector[C] prob_joint;\n  vector[J] log_lik;\n  array[C] real prob_attr_class;\n  matrix[J,I] y_rep;\n\n  for (j in 1:J){{\n   for (c in 1:C){{\n     for(i in 1:I){{\n       real p = fmin(fmax(pi[i,c], 1e-9), (1 - 1e-9));\n       eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);\n     }}\n     prob_joint[c] = exp(log_nu[c]) * exp(sum(eta));\n     log_lik[j] = log_sum_exp(prob_joint);\n   }}\n   prob_resp_class[j] = prob_joint/sum(prob_joint);\n  }}\n\n  for (j in 1:J){{\n    for (k in 1:K){{\n      for (c in 1:C){{\n        prob_attr_class[c] = prob_resp_class[j,c] * alpha[c,k];\n      }}\n      prob_resp_attr[j,k] = sum(prob_attr_class);\n    }}\n  }}\n\n  for (j in 1:J){{\n    int z = categorical_rng(nu);\n    for (i in 1:I){{\n      y_rep[j,i] = bernoulli_rng(pi[i,z]);\n    }}\n  }}\n}}'

    return quant

  @reactive.event(input.build_model)
  def stan_generateprior():
    prof_model = input.prof_model()
    
    if prof_model == 'linear':
      quant = f'generated quantities {{\n  matrix[J,I] y_rep;\n\n  for (j in 1:J){{\n    int z = categorical_rng(nu/sum(nu));\n    for (i in 1:I){{\n      y_rep[j,i] = bernoulli_rng(pi[i,z]);\n    }}\n  }}\n}}'
    
    else:
      quant = f'generated quantities {{\n  matrix[J,I] y_rep;\n\n  for (j in 1:J){{\n    int z = categorical_rng(nu);\n    for (i in 1:I){{\n      y_rep[j,i] = bernoulli_rng(pi[i,z]);\n    }}\n  }}\n}}'
  
    return quant
  
  @reactive.event(input.build_model)
  def stan_paramchunk():
    q = loaded_q()
    # attr_num = q.shape[1] - 1
    attr_num = q.filter(regex = 'attr').shape[1]
    prof_model = input.prof_model()
    
    if prof_model == 'linear':
      param = 'parameters {\n  vector<lower=0, upper=1>[I] slip;\n  vector<lower=0, upper=1>[I] guess;\n'
      
      if attr_num == 2:
        attr_list = [1, 2]
      elif attr_num == 3:
        attr_list = [1, 2, 3]
      elif attr_num == 4:
        attr_list = [1, 2, 3, 4]
      elif attr_num == 5:
        attr_list = [1, 2, 3, 4, 5]

      lambdas = '\n'.join([f'  real<lower=0, upper=1> lambda{i};' for i in attr_list])
      
      return param + lambdas + '\n}'
    
    elif prof_model == 'unconstrained':
      param = 'parameters {\n  simplex[C] nu;\n  vector<lower=0, upper=1>[I] slip;\n  vector<lower=0, upper=1>[I] guess;'
      
      return param + '\n}'
  
  @reactive.event(input.build_model)
  def stan_tparamchunk():
    q = loaded_q()
    # attr_num = q.shape[1] - 1
    attr_num = q.filter(regex = 'attr').shape[1]
    prof_model = input.prof_model()
    
    if prof_model == 'linear':
      param = 'transformed parameters {\n  vector[C] nu;\n  vector[C] log_nu;\n  matrix[I,C] pi;\n'
      
      attr1_line = '\n  for (c in 1:C) {\n    real theta1 = (alpha[c,1] == 1) ? lambda1 : (1 - lambda1);\n'
      
      pi_calc = '\n  log_nu = log(nu);\n\n  for (c in 1:C){\n    for (i in 1:I){\n      pi[i,c] = pow((1 - slip[i]), xi[i,c]) *\n      pow(guess[i], (1 - xi[i,c]));\n    }\n  }\n}'
      
      if attr_num == 2:
        more_attr_line = '\n    real theta2;\n\n    if (alpha[c,1] == 1) {\n      theta2 = (alpha[c,2] == 1) ? lambda2 : (1 - lambda2);\n    } \n    else {\n      theta2 = (alpha[c,2] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    nu[c] = theta1 * theta2;\n  }\n'
        
        trans_param = param + attr1_line + more_attr_line + pi_calc
        
        return trans_param
      
      elif attr_num == 3:
        more_attr_line = '\n    real theta2;\n    real theta3;\n\n    if (alpha[c,1] == 1) {\n      theta2 = (alpha[c,2] == 1) ? lambda2 : (1 - lambda2);\n    } \n    else {\n      theta2 = (alpha[c,2] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    if (alpha[c,1] == 1 && alpha[c,2] == 1) {\n      theta3 = (alpha[c,3] == 1) ? lambda3 : (1 - lambda3);\n    }\n    else {\n      theta3 = (alpha[c,3] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    nu[c] = theta1 * theta2 * theta3;\n  }\n'

        trans_param = param + attr1_line + more_attr_line + pi_calc
        
        return trans_param
      
      elif attr_num == 4:
        more_attr_line = '\n    real theta2;\n    real theta3;\n    real theta4;\n\n    if (alpha[c,1] == 1) {\n      theta2 = (alpha[c,2] == 1) ? lambda2 : (1 - lambda2);\n    } \n    else {\n      theta2 = (alpha[c,2] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    if (alpha[c,1] == 1 && alpha[c,2] == 1) {\n      theta3 = (alpha[c,3] == 1) ? lambda3 : (1 - lambda3);\n    }\n    else {\n      theta3 = (alpha[c,3] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    if (alpha[c,1] == 1 && alpha[c,2] == 1 && alpha[c,3] == 1) {\n      theta4 = (alpha[c,4] == 1) ? lambda4 : (1 - lambda4);\n    }\n    else {\n      theta4 = (alpha[c,4] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    nu[c] = theta1 * theta2 * theta3 * theta4;\n  }\n'

        trans_param = param + attr1_line + more_attr_line + pi_calc
        
        return trans_param
      
      elif attr_num == 5:
        more_attr_line = '\n    real theta2;\n    real theta3;\n    real theta4;\n    real theta5;\n\n    if (alpha[c,1] == 1) {\n      theta2 = (alpha[c,2] == 1) ? lambda2 : (1 - lambda2);\n    } \n    else {\n      theta2 = (alpha[c,2] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    if (alpha[c,1] == 1 && alpha[c,2] == 1) {\n      theta3 = (alpha[c,3] == 1) ? lambda3 : (1 - lambda3);\n    }\n    else {\n      theta3 = (alpha[c,3] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    if (alpha[c,1] == 1 && alpha[c,2] == 1 && alpha[c,3] == 1) {\n      theta4 = (alpha[c,4] == 1) ? lambda4 : (1 - lambda4);\n    }\n    else {\n      theta4 = (alpha[c,4] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    if (alpha[c,1] == 1 && alpha[c,2] == 1 && alpha[c,3] == 1 && alpha[c,4] == 1) {\n      theta5 = (alpha[c,5] == 1) ? lambda5 : (1 - lambda5);\n    }\n    else {\n      theta5 = (alpha[c,5] == 1) ? 1e-9 : (1 - 1e-9);\n    }\n    nu[c] = theta1 * theta2 * theta3 * theta4 * theta5;\n  }\n'

        trans_param = param + attr1_line + more_attr_line + pi_calc
        
        return trans_param
        
    elif prof_model == 'unconstrained':
      param = 'transformed parameters {\n  vector[C] log_nu;\n  matrix[I,C] pi;\n'
      
      pi_calc = '\n  log_nu = log(nu);\n\n  for (c in 1:C){\n    for (i in 1:I){\n      pi[i,c] = pow((1 - slip[i]), xi[i,c]) *\n      pow(guess[i], (1 - xi[i,c]));\n    }\n  }\n}'
      
      trans_param = param + pi_calc
      
      return trans_param
      
  @reactive.event(input.build_model)
  def stan_modelchunk():
    q = loaded_q()
    # n = q.shape[1] - 1
    n = q.filter(regex = 'attr').shape[1]
    prof_model = input.prof_model()
    
    slip_alpha = input.slip_alpha()
    slip_beta = input.slip_beta()
    guess_alpha = input.guess_alpha()
    guess_beta = input.guess_beta()
  
    model_start = f'model{{\n  array[C] real ps;\n  array[I] real eta;\n\n  for (i in 1:I){{\n    slip[i] ~ beta({slip_alpha}, {slip_beta});\n    guess[i] ~ beta({guess_alpha}, {guess_beta});\n  }}\n'
    
    if prof_model == 'linear':
      
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

    model_end = f'  for (j in 1:J){{\n    for (c in 1:C){{\n      for (i in 1:I){{\n        real p = fmin(fmax(pi[i,c], 1e-9), (1 - 1e-9));\n        eta[i] = Y[j,i] * log(p) + (1 - Y[j,i]) * log1m(p);\n      }}\n      ps[c] = log_nu[c] + sum(eta);\n    }}\n    target += log_sum_exp(ps);\n  }}\n}}'
  
    if prof_model == 'linear':
      model = model_start + priors + model_end
    
    elif prof_model == 'unconstrained':
      model = model_start + model_end

    return model
  
  @reactive.event(input.build_model)
  def stan_priormodelchunk():
    q = loaded_q()
    # n = q.shape[1] - 1
    n = q.filter(regex = 'attr').shape[1]
    prof_model = input.prof_model()
    
    slip_alpha = input.slip_alpha()
    slip_beta = input.slip_beta()
    guess_alpha = input.guess_alpha()
    guess_beta = input.guess_beta()
  
    model_start = f'model{{\n  array[C] real ps;\n  array[I] real eta;\n\n  for (i in 1:I){{\n    slip[i] ~ beta({slip_alpha}, {slip_beta});\n    guess[i] ~ beta({guess_alpha}, {guess_beta});\n  }}\n'
    
    if prof_model == 'linear':
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
      
    if prof_model == 'linear':
      model = model_start + priors + '}'
    
    elif prof_model == 'unconstrained':
      model = model_start + '}'

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
          
          p.set(3,
                detail = 'Running model.')
          
          pfit = prior_model.sample(
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
          
          p.set(3,
                detail = 'Running model.')
          
          pfit = prior_model.sample(
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
    
    return pd.DataFrame(fit.summary())

  @reactive.effect
  @reactive.event(input.run_model)
  def save_diagnostics():
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
    
    summary_df.to_csv(diag_dir/'model_diagnostics.csv')
    prior_df.to_csv(diag_dir/'prior_diagnostics.csv')

  @reactive.effect
  @reactive.event(input.run_model)
  def save_data_models_fits():
    df = loaded_data()
    q = loaded_q()
    # n = q.shape[1] - 1
    n = q.filter(regex = 'attr').shape[1]
    alpha = create_alpha()
    xi = create_xi()
    
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
    pd.DataFrame(xi).to_csv(f'{job_dir}/xi.csv')
    
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
  
  @render.download(filename = 'dcm_report.pdf')
  def download_report():
    if model_fit.get() is None or prior_fit.get() is None:
      raise SafeException('Run the model first so report inputs exist.')

    with ui.Progress(min = 0, max = 100) as p:
      p.set(message='Rendering report...', detail = 'Preparing parameters', value = 10)
      
      report_dir = Path(here('report'))
      qmd_path = report_dir / 'report.qmd'
      out_name = 'dcm_report.pdf'

      if not qmd_path.exists():
        raise SafeException(f'{qmd_path} was not found.')

      cmd = [
        'quarto', 'render', str(qmd_path),
        '--to', 'typst',
        '-P', f'threshold:{input.threshold()}',
        '--output', out_name
        ]
      
      p.set(detail = 'Generating document with Quarto...', value = 50)
      
      env = os.environ.copy()
      env['QUARTO_PYTHON'] = sys.executable
      
      result = subprocess.run(
        cmd,
        capture_output = True,
        text = True,
        env = env,
        cwd = str(report_dir)
        )
      
      if result.returncode != 0:
          raise SafeException(result.stderr.strip() or 'Quarto render failed.')
      
      p.set(detail = 'Done! Document is downloading.', value = 95)
      
      yield (report_dir / out_name).read_bytes()

# --------------------------------------------------------------------------------------------------------

app = App(app_ui, server)
