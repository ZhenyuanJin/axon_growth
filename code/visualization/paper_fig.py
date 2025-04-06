import sys
sys.path.append('../utils_function/')
sys.path.append(r'/data/zyjin/common_func/')
import utils_function as uf
import common_functions as cf
import pandas as pd
import numpy as np
import os
import pdb


paper_fig = uf.PaperFig(model_timedir_list=['../../results/optimal_results/mouse_gamanut/2025_04_01_22_01_11', '../../results/optimal_results/marmoset_majka/2025_04_01_22_07_20', '../../results/optimal_results/macaque_markov40/2025_04_01_22_18_12', '../../results/optimal_results/human_lariviere/2025_04_01_22_26_53'])
# paper_fig.fig_1()
paper_fig.fig_2()
paper_fig.fig_3()
paper_fig.fig_4()
paper_fig.fig_5()
paper_fig.fig_6()
paper_fig.fig_7()
paper_fig.fig_8()
paper_fig.SI_fig_1()
paper_fig.SI_fig_2()
paper_fig.SI_fig_3()
paper_fig.SI_fig_4()
paper_fig.SI_fig_5()
paper_fig.SI_fig_6()
paper_fig.SI_fig_7()
paper_fig.SI_fig_8()
paper_fig.SI_fig_9()
paper_fig.SI_fig_10()
paper_fig.table_1()