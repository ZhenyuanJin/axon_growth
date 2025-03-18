# region import
# 标准库导入
import subprocess
import os
import re
import sys
import json
import pickle
import joblib
import random
import shutil
import time
import datetime
import warnings
from math import ceil
import multiprocessing
from multiprocessing import Process
from pathlib import Path
import inspect
import types
from functools import wraps, partial
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
import copy
import hashlib
import json
import logging
from cProfile import Profile
from pstats import Stats
import tracemalloc


# 数学和科学计算库
import numpy as np
import math
import scipy
import scipy.stats as st
from scipy.stats import gaussian_kde, zscore
from scipy.integrate import quad
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sps
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA, NMF
import statsmodels
from statsmodels.tsa.stattools import acf, acovf, ccf
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.smoothers_lowess import lowess


# 数据处理和可视化库
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent, LogFormatterMathtext, LogFormatterSciNotation, PercentFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm, Normalize, ListedColormap, SymLogNorm, LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from pdf2image import convert_from_path
import imageio
import shapely.geometry
import pointpats


# 自定义库
sys.path.append(r'/data/zyjin/common_func/')
import common_functions as cf
# endregion


# region global_variable
SOURCE_COLOR = cf.RED
TARGET_COLOR = cf.GREEN
# endregion


# region pre_process
def coords_to_polygon(x, y):
    '''
    将coords转换为polygon
    '''
    return shapely.geometry.Polygon(np.column_stack((x, y)))


def organize_txt_coords(filename):
    '''
    将txt格式的coords数据整理
    '''
    name_position_start = str(filename).rfind('/')
    name_position_end = str(filename).find('.txt')
    name = str(filename)[name_position_start+1: name_position_end]

    txt_content = cf.load_txt(filename)
    txt_content = [l for l in txt_content if l != ''] # 去掉空行
    txt_content = [l.split() for l in txt_content] # 按空格分割
    txt_content = [[float(item) for item in l] for l in txt_content] # 转为float
    txt_content = np.array(txt_content) # 转为np.array
    vx, vy = txt_content[:, 0], txt_content[:, 1]

    coords = np.array([np.array([i,j]) for i, j in zip(vx, vy)])
    polygon = coords_to_polygon(vx, vy)
    results = {'coords': coords, 'polygon': polygon, 'name': name}
    return results


def generate_coords_polygon(animal, dataset_key):
    '''
    生成coords的polygon数据,并且保存
    '''
    animal_dir = AnimalDir(animal, dataset_key)

    cf.mkdir(animal_dir.area_coords_dir)

    # 获取每个area
    coords_dict = {}
    polygon_dict = {}
    for filename in cf.get_subfile(animal_dir.raw_area_coords_dir):
        r = organize_txt_coords(filename)
        name = r['name']
        coords = r['coords']
        polygon = r['polygon']
        coords_dict[name] = coords
        polygon_dict[name] = polygon
    cf.save_dict(coords_dict, animal_dir.area_coords_dir + 'coords_dict')
    cf.save_dict(polygon_dict, animal_dir.area_coords_dir + 'polygon_dict')

    # 获取whole_cortex
    for filename in cf.get_subfile(animal_dir.raw_whole_cortex_coords_dir):
        r = organize_txt_coords(filename)
        whole_cortex_coords, whole_cortex_polygon = r['coords'], r['polygon']
    results = {'coords_dict': coords_dict, 'polygon_dict': polygon_dict, 'whole_cortex_coords': whole_cortex_coords, 'whole_cortex_polygon': whole_cortex_polygon}
    return results


def generate_area_center(animal, dataset_key, density=100):
    '''
    获取脑区中心坐标(在内部随机生成点,获取重心)
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    polygon_dict = animal_info.polygon_dict
    
    # 固定随机种子
    seed = 0

    area_center_dict = {}
    for area, polygon in polygon_dict.items():
        points = generate_random_point_in_polygon(polygon=polygon, density=density, seed=seed)
        cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        area_center_dict[area] = np.array([cx,cy])
    cf.save_dict(area_center_dict, animal_info.area_coords_dir + 'area_center_dict')
    return area_center_dict


def generate_name_location(animal, dataset_key):
    '''
    生成脑区名字和位置的对应关系(此位置保证name在polygon内部,且较美观)
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    
    area_center_dict = generate_area_center(animal, dataset_key)
    name_location_dict = area_center_dict.copy() # 先凑合着用
    cf.save_dict(name_location_dict, animal_info.area_coords_dir + 'name_location_dict')
    return name_location_dict


def generate_angle_matrix(animal, dataset_key, dim='2D'):
    '''
    生成
    '''
    pass


def generate_distance_matrix(animal, dataset_key, dim='2D'):
    '''
    生成距离矩阵
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    if dim == '2D':
        area_center_dict = animal_info.area_center_dict
        full_split_name = animal_info.full_split_name
        split_distance_matrix = cf.get_zero_df(full_split_name, full_split_name)
        for i, center_i in area_center_dict.items():
            for j, center_j in area_center_dict.items():
                split_distance_matrix.loc[i, j] = np.linalg.norm(center_i - center_j, ord=2)
        distance_matrix = convert_split_df_to_normal(split_distance_matrix, animal, dataset_key, method='mean')
        cf.save_pkl(distance_matrix, animal_info.distance_matrix_dir + 'distance_matrix_2D')
    else:
        raise NotImplementedError
    return distance_matrix
# endregion


# region load_data
class AnimalDir:
    def __init__(self, animal, dataset_key):
        self.animal = animal
        self.dataset_key = dataset_key
        self.animal_and_dataset_key = str(animal) + '_' + str(dataset_key)

        # 数据目录
        self.raw_data_dir = '../../raw_data/' + self.animal_and_dataset_key + '/'
        self.data_dir = '../../data/' + self.animal_and_dataset_key + '/'
        self.raw_area_coords_dir = self.raw_data_dir + 'area_coords/'
        self.raw_whole_cortex_coords_dir = self.raw_data_dir + 'whole_cortex_coords/'
        self.area_coords_dir = self.data_dir + 'area_coords/'
        self.whole_cortex_coords_dir = self.data_dir + 'whole_cortex_coords/'
        self.distance_matrix_dir = self.data_dir + 'distance_matrix/'

        # 结果目录
        self.axon_source_target_dir = 'pass'



class AnimalInfo(AnimalDir):
    def __init__(self, animal, dataset_key):
        '''
        对于基础信息,直接读取;对于要计算的信息,先尝试读取,如果没有,则计算
        '''
        super().__init__(animal, dataset_key)

        SC_type_dict = {'mouse_gamanut': 'FLN', 'marmoset_majka': 'FLN', 'marmoset_V4': 'FLN', 'macaque_markov29': 'FLN', 'macaque_markov40': 'FLN', 'human_lariviere': 'Dmri'}
        self.SC_type = SC_type_dict[self.animal_and_dataset_key]

        name_information = get_names(self.animal, self.dataset_key)
        self.source_name = name_information['source_name']
        self.target_name = name_information['target_name']
        self.full_name = name_information['full_name']
        self.full_split_name = name_information['full_split_name']
        self.name_to_split_name_dict = name_information['name_to_split_name_dict']
        self.split_name_to_name_dict = name_information['split_name_to_name_dict']
        self.column_mode = 'target' # never be source, this project is always target



        # self.category_dict = load_category_dict(self.animal, self.dataset_key, category_mode=...)

    @property
    def reordered_full_name(self):
        if self.SC_focus_direction == 'row':
            return prioritize_sublist(self.full_name, self.source_name)
        elif self.SC_focus_direction == 'col':
            return prioritize_sublist(self.full_name, self.target_name)

    @property
    def coords_dict(self):
        try:
            return load_coords_dict(self.animal, self.dataset_key)
        except:
            return generate_coords_polygon(self.animal, self.dataset_key)['coords_dict']
    
    @property
    def polygon_dict(self):
        try:
            return load_polygon_dict(self.animal, self.dataset_key)
        except:
            return generate_coords_polygon(self.animal, self.dataset_key)['polygon_dict']

    @property
    def whole_cortex_coords(self):
        try:
            return load_whole_cortex_coords(self.animal, self.dataset_key)
        except:
            return generate_coords_polygon(self.animal, self.dataset_key)['whole_cortex_coords']

    @property
    def whole_cortex_polygon(self):
        try:
            return load_whole_cortex_polygon(self.animal, self.dataset_key)
        except:
            return generate_coords_polygon(self.animal, self.dataset_key)['whole_cortex_polygon']

    @property
    def area_center_dict(self):
        try:
            return load_area_center_dict(self.animal, self.dataset_key)
        except:
            return generate_area_center(self.animal, self.dataset_key)

    @property
    def name_location_dict(self):
        try:
            return load_name_location_dict(self.animal, self.dataset_key)
        except:
            return generate_name_location(self.animal, self.dataset_key)

    @property
    def angle_2D(self):
        pass

    @property
    def angle_3D(self):
        pass

    @property
    def distance_2D(self):
        pass

    @property
    def distance_3D(self):
        pass

    @property
    def neuron_density(self):
        return load_neuron_density(self.animal, self.dataset_key)
    
    @property
    def connection_matrix_vmax(self):
        '''
        可视化连接矩阵时的最大值
        '''
        return 0.2

    @property
    def c(self):
        '''
        log时的平移系数
        '''
        if self.animal == 'human':
            return 6
        elif self.animal == 'marmoset' and self.dataset_key == 'Dmri_Tian':
            raise NotImplementedError
        else:
            return 7

    @property
    def w(self):
        '''
        当SC为0时,设定的基准值
        '''
        return 0.9
    
    @property
    def SC_focus_direction(self):
        '''
        当分析的时候,关注row还是col
        '''
        row_animal_and_dataset_key_list = []
        col_animal_and_dataset_key_list = ['mouse_gamanut', 'marmoset_majka', 'marmoset_V4', 'macaque_markov29', 'macaque_markov40', 'human_lariviere']
        if self.animal_and_dataset_key in row_animal_and_dataset_key_list:
            return 'row'
        elif self.animal_and_dataset_key in col_animal_and_dataset_key_list:
            return 'col'
    
    @property
    def SC_data(self):
        '''
        读取SC数据
        '''
        pass
        


def load_coords_dict(animal, dataset_key):
    '''
    读取coords数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'coords_dict')


def load_polygon_dict(animal, dataset_key):
    '''
    读取polygon数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'polygon_dict')


def load_whole_cortex_coords(animal, dataset_key):
    '''
    读取whole_cortex_coords数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'whole_cortex_coords')


def load_whole_cortex_polygon(animal, dataset_key):
    '''
    读取whole_cortex_polygon数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'whole_cortex_polygon')


def load_area_center_dict(animal, dataset_key):
    '''
    读取area_center数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'area_center_dict')


def load_name_location_dict(animal, dataset_key):
    '''
    读取name_location数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'name_location_dict')


def load_SC(filename):
    '''
    读取SC的基本数据
    '''
    SC = cf.load_df(filename)
    return SC

@cf.to_be_improved
def load_neuron_density(animal, dataset_key):
    '''
    读取neuron_density数据
    '''
    animal_dir = AnimalDir(animal, dataset_key)


    animal_info = AnimalInfo(animal, dataset_key)
    return cf.create_dict(animal_info.full_name, 1)


@cf.to_be_improved
def get_names(animal, dataset_key):
    '''
    Note that for SC matrix, the name of each area is in full_name. However, for coords, they may be divided into several parts, for example, 11--1 and 11--2. Coords names are all_name_coords in code. Sometimes, you need to transform to another type, just use accurate_name_dict and name2namelist_dic
    '''
    animal_and_dataset_key = animal + '_' + dataset_key

    if animal_and_dataset_key == 'macaque_markov29':
        # target_name = ['V1', 'V2', 'V4', 'TEO', 'MT', 'Tepd', 'DP', 'STPc', '7A', 'STPr', 'STPi', 'PBr', '8l', '7m', '46d', '8m', '7B', '946d', '946v', '8B', '10', 'F7', 'F5', '5', 'F2',  '24c', 'ProM', '2', 'F1']
        target_name = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', 'TEO', '2', 'F1', 'STPc', '7A', '46d', '10', '946v', '946d', 'F5', 'Tepd', 'PBr', '7m', '7B', 'F2', 'STPi', 'ProM', 'F7', '8B', 'STPr', '24c']
        coords_target_name = target_name
        source_name = ['10', '11', '12', '13', '14', '1', '23', '24a', '24b', '24c', '24d', '25', '2930', '2', '31', '32', '3', '44', '45A', '45B', '46d', '46v', '5', '7A', '7B', '7m', '7op', '8B', '8l', '8m', '8r', '946d', '946v', '9', 'AIP', 'CORE', 'DP', 'ENTORHINAL', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'FST', 'Gu', 'INSULA', 'IPa', 'LB', 'LIP', 'MB', 'MIP', 'MST', 'MT', 'OPAI', 'OPRO', 'PBc', 'PBr', 'PERIRHINAL', 'PGa', 'PIP', 'PIRIFORM', 'Pi', 'Pro.St.', 'ProM', 'SII', 'STPc', 'STPi', 'STPr', 'SUBICULUM', 'TEMPORAL_POLE', 'TEO', 'TEOm', 'TEad', 'TEama', 'TEamp', 'TEav', 'TEpv', 'THTF', 'TPt', 'Tepd', 'V1', 'V2', 'V3A', 'V3', 'V4', 'V4t', 'V6A', 'V6', 'VIP']
        accurate_name_dict = {}
        for name in all_name_coords:
            if name in ['11--1', '11--2']:
                accurate_name_dict[name] = '11'
            else:
                accurate_name_dict[name] = name
        name2namelist_dict = {}
        for name in source_name:
            if name in ['11']:
                name2namelist_dict[name] = ['11--1', '11--2']
            else:
                name2namelist_dict[name] = [name]

    if animal_and_dataset_key == 'macaque_markov40':
        target_name = ['V1', 'V2', 'V4', 'TEO', '946d', 'F5', '8m', '7A', 'DP', '2', '5', '7B', 'STPr', 'STPi', 'STPc', 'PBr', 'Tepd', '24c', 'F1', 'F2', 'F7', 'ProM', '8l', '946v', '46d', '8B', '10', 'MT', '7m', '9', '25', 'F3', 'F4', '32', 'LIP', 'OPRO', '3', '1', '45A', 'V6']
        coords_target_name = target_name
        source_name = ['10', '11', '12', '13', '14', '1', '23', '24a', '24b', '24c', '24d', '25', '2930', '2', '31', '32', '3', '44', '45A', '45B', '46d', '46v', '5', '7A', '7B', '7m', '7op', '8B', '8l', '8m', '8r', '946d', '946v', '9', 'AIP', 'CORE', 'DP', 'ENTORHINAL', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'FST', 'Gu', 'INSULA', 'IPa', 'LB', 'LIP', 'MB', 'MIP', 'MST', 'MT', 'OPAI', 'OPRO', 'PBc', 'PBr', 'PERIRHINAL', 'PGa', 'PIP', 'PIRIFORM', 'Pi', 'Pro.St.', 'ProM', 'SII', 'STPc', 'STPi', 'STPr', 'SUBICULUM', 'TEMPORAL_POLE', 'TEO', 'TEOm', 'TEad', 'TEama', 'TEamp', 'TEav', 'TEpv', 'THTF', 'TPt', 'Tepd', 'V1', 'V2', 'V3A', 'V3', 'V4', 'V4t', 'V6A', 'V6', 'VIP']
        accurate_name_dict = {}
        for name in all_name_coords:
            if name in ['11--1', '11--2']:
                accurate_name_dict[name] = '11'
            else:
                accurate_name_dict[name] = name
        name2namelist_dict = {}
        for name in source_name:
            if name in ['11']:
                name2namelist_dict[name] = ['11--1', '11--2']
            else:
                name2namelist_dict[name] = [name]

    if animal_and_dataset_key == 'marmoset_majka':
        target_name = ['A1-2', 'A10', 'A11', 'A19DI', 'A19M', 'A23a', 'A23b', 'A23c', 'A24d', 'A32', 'A32V', 'A3a', 'A3b', 'A45', 'A46D', 'A47L', 'A4ab', 'A4c', 'A6DC', 'A6DR', 'A6M', 'A6Va', 'A8C', 'A8aD', 'A8aV', 'A8b', 'A9', 'AIP', 'AuA1', 'AuCM', 'AuCPB', 'AuML', 'AuRT', 'LIP', 'MIP', 'MST', 'OPt', 'PE', 'PEC', 'PF', 'PFG', 'PG', 'PGM', 'PGa-IPa', 'S2E', 'TE3', 'TEO', 'TPO', 'V1', 'V2', 'V3A', 'V4', 'V4T', 'V5', 'V6']
        # coords_target_name = ['A1-2', 'A10', 'A11', 'A19DI', 'A19M', 'A23a', 'A23b', 'A23c--1', 'A23c--2', 'A24d', 'A32', 'A32V', 'A3a', 'A3b', 'A45', 'A46D', 'A47L', 'A4ab', 'A4c', 'A6DC', 'A6DR', 'A6M', 'A6Va', 'A8C', 'A8aD', 'A8aV', 'A8b', 'A9', 'AIP', 'AuA1', 'AuCM', 'AuCPB', 'AuML', 'AuRT', 'LIP', 'MIP', 'MST', 'OPt', 'PE', 'PEC', 'PF', 'PFG', 'PG', 'PGM', 'PGa-IPa', 'S2E', 'TE3', 'TEO', 'TPO', 'V1', 'V2--1', 'V2--2', 'V3A', 'V4', 'V4T', 'V5', 'V6']
        source_name = ['A1-2', 'A10', 'A11', 'A13L', 'A13M', 'A13a', 'A13b', 'A14C', 'A14R', 'A19DI', 'A19M', 'A23V', 'A23a', 'A23b', 'A23c', 'A24a', 'A24b', 'A24c', 'A24d', 'A25', 'A29a-c', 'A29d', 'A30', 'A31', 'A32', 'A32V', 'A35', 'A36', 'A3a', 'A3b', 'A45', 'A46D', 'A46V', 'A47L', 'A47M', 'A47O', 'A4ab', 'A4c', 'A6DC', 'A6DR', 'A6M', 'A6Va', 'A6Vb', 'A8C', 'A8aD', 'A8aV', 'A8b', 'A9', 'AI', 'AIP', 'APir', 'AuA1', 'AuAL', 'AuCL', 'AuCM', 'AuCPB', 'AuML', 'AuR', 'AuRM', 'AuRPB', 'AuRT', 'AuRTL', 'AuRTM', 'DI', 'Ent', 'FST', 'GI', 'Gu', 'IPro', 'LIP', 'MIP', 'MST', 'OPAl', 'OPro', 'OPt', 'PE', 'PEC', 'PF', 'PFG', 'PG', 'PGM', 'PGa-IPa', 'PaIL', 'PaIM', 'Pir', 'ProM', 'ProSt', 'ReI', 'S2E', 'S2I', 'S2PR', 'S2PV', 'STR', 'TE1', 'TE2', 'TE3', 'TEO', 'TF', 'TFO', 'TH', 'TL', 'TLO', 'TPO', 'TPPro', 'TPro', 'TPt', 'V1', 'V2', 'V3', 'V3A', 'V4', 'V4T', 'V5', 'V6', 'V6A', 'VIP']
        full_name = cf.union_list(target_name, source_name)
        name_to_split_name_dict = {}
        for name in full_name:
            name_to_split_name_dict[name] = [name]
            if name in ['A23c']:
                name_to_split_name_dict[name] = ['A23c--1', 'A23c--2']
            if name in ['A14C']:
                name_to_split_name_dict[name] = ['A14C--1', 'A14C--2']
            if name in ['A14R']:
                name_to_split_name_dict[name] = ['A14R--1', 'A14R--2']
            if name in ['A23V']:
                name_to_split_name_dict[name] = ['A23V--1', 'A23V--2']
            if name in ['A24a']:
                name_to_split_name_dict[name] = ['A24a--1', 'A24a--2']
            if name in ['A25']:
                name_to_split_name_dict[name] = ['A25--1', 'A25--2']
            if name in ['A24a-c']:
                name_to_split_name_dict[name] = ['A24a-c--1', 'A24a-c--2']
            if name in ['A29a-c']:
                name_to_split_name_dict[name] = ['A29a-c--1', 'A29a-c--2']
            if name in ['A29d']:
                name_to_split_name_dict[name] = ['A29d--1', 'A29d--2']
            if name in ['A30']:
                name_to_split_name_dict[name] = ['A30--1', 'A30--2']
            if name in ['A35']:
                name_to_split_name_dict[name] = ['A35--1', 'A35--2']
            if name in ['ProSt']:
                name_to_split_name_dict[name] = ['ProSt--1', 'ProSt--2', 'ProSt--3']
            if name in ['TH']:
                name_to_split_name_dict[name] = ['TH--1', 'TH--2']
            if name in ['V2']:
                name_to_split_name_dict[name] = ['V2--1', 'V2--2']
        # full_split_name = cf.union_list(*name_to_split_name_dict.values())
        full_split_name = []
        for name, name_to_split_name in name_to_split_name_dict.items():
            full_split_name.extend(name_to_split_name)
        split_name_to_name_dict = {}
        for name in full_split_name:
            split_name_to_name_dict[name] = name
            if name in ['A23c--1', 'A23c--2']:
                split_name_to_name_dict[name] = 'A23c'
            if name in ['A14C--1', 'A14C--2']:
                split_name_to_name_dict[name] = 'A14C'
            if name in ['A14R--1', 'A14R--2']:
                split_name_to_name_dict[name] = 'A14R'
            if name in ['A23V--1', 'A23V--2']:
                split_name_to_name_dict[name] = 'A23V'
            if name in ['A24a--1', 'A24a--2']:
                split_name_to_name_dict[name] = 'A24a'
            if name in ['A25--1', 'A25--2']:
                split_name_to_name_dict[name] = 'A25'
            if name in ['A24a-c--1', 'A24a-c--2']:
                split_name_to_name_dict[name] = 'A24a-c'
            if name in ['A29a-c--1', 'A29a-c--2']:
                split_name_to_name_dict[name] = 'A29a-c'
            if name in ['A29d--1', 'A29d--2']:
                split_name_to_name_dict[name] = 'A29d'
            if name in ['A30--1', 'A30--2']:
                split_name_to_name_dict[name] = 'A30'
            if name in ['A35--1', 'A35--2']:
                split_name_to_name_dict[name] = 'A35'
            if name in ['ProSt--1', 'ProSt--2', 'ProSt--3']:
                split_name_to_name_dict[name] = 'ProSt'
            if name in ['TH--1', 'TH--2']:
                split_name_to_name_dict[name] = 'TH'
            if name in ['V2--1', 'V2--2']:
                split_name_to_name_dict[name] = 'V2'

    if animal_and_dataset_key == 'marmoset_riken':
        target_name = ['A10', 'A9', 'A8b', 'A46D', 'A46V', 'A8aD', 'A8aV', 'A47L', 'A47M', 'A45', 'A47O', 'ProM', 'A11', 'A13a', 'A13b', 'A13L', 'A13M', 'OPAl', 'OPro', 'Gu', 'A14C', 'A14R', 'A25', 'A32V', 'A32', 'A24a', 'A24b', 'A24c', 'A24d', 'A6DR', 'A6DC', 'A6Va', 'A6Vb', 'A6M', 'A8C', 'A4ab', 'A4c', 'AI', 'DI', 'GI', 'ReI', 'IPro', 'PaIL', 'PaIM', 'TPro', 'A3a', 'A3b', 'A1_2', 'S2E', 'S2I', 'S2PR', 'S2PV', 'AuRTL', 'AuRT', 'AuRPB', 'AuRTM', 'AuR', 'AuRM', 'AuAL', 'AuA1', 'AuCM', 'AuCPB', 'AuML', 'AuCL', 'STR', 'TPt', 'TE1', 'TE2', 'TE3', 'TEO', 'Pga_IPa', 'TPO', 'TPPro', 'Ent', 'A36', 'A35', 'TF', 'TL', 'TH', 'TLO', 'TFO', 'PaS', 'PrS', 'A23a', 'A23b', 'A23c', 'A23V', 'A29a_c', 'A29d', 'A30', 'ProSt', 'PGM', 'A31', 'PE', 'PEC', 'V6A', 'PF', 'PFG', 'OPt', 'PG', 'VIP', 'MIP', 'AIP', 'LIP', 'V1', 'V2', 'V3', 'V3A', 'A19DI', 'A19M', 'V4', 'V4T' 'V5', 'MST', 'FST', 'V6', 'Pir']
        source_name = ['A11', 'A45', 'A13M', 'A8b', 'A32', 'A24a', 'A8aD', 'A46V', 'A8aV', 'A46D', 'A6Vb', 'A24b', 'A6DR', 'A25', 'A14R', 'A13L', 'A6M', 'A9', 'A10', 'A47O']
        coords_source_name = []
        accurate_name_dict = {}
        for name in all_name_coords:
            accurate_name_dict[name] = name
            pass
        name2namelist_dict = {}
        for name in source_name:
            name2namelist_dict[name] = [name]
            pass

    if animal_and_dataset_key == 'marmoset-V4':
        target_name = ['4', '9', '10', '12', '14', '15', '18', '19', '20', '21', '25', '27', '28', '29', '31', '33', '34', '35', '39', '40', '41', '45', '46', '48', '49', '50', '58', '62', '64', '66', '67', '72', '73', '74', '79', '81', '88', '91', '92', '94', '95']
        source_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96']
        coords_target_name = ['4', '9', '10', '12', '14', '15', '18', '19', '20', '21', '25', '27', '28', '29', '31', '33', '34', '35', '39', '40', '41', '45', '46', '48', '49', '50', '58', '62', '64', '66', '67', '72', '73', '74', '79', '81', '88', '91', '92', '94', '95']
        accurate_name_dict = {}
        for name in all_name_coords:
            accurate_name_dict[name] = name
            if name in ['11--1', '11--2']:
                accurate_name_dict[name] = '11'
            if name in ['43--1', '43--2']:
                accurate_name_dict[name] = '43'
            if name in ['77--1', '77--2']:
                accurate_name_dict[name] = '77'
        name2namelist_dict = {}
        for name in source_name:
            name2namelist_dict[name] = [name]
            if name in ['11']:
                name2namelist_dict[name] = ['11--1', '11--2']
            if name in ['43']:
                name2namelist_dict[name] = ['43--1', '43--2']
            if name in ['77']:
                name2namelist_dict[name] = ['77--1', '77--2']

    if animal_and_dataset_key == 'mouse':
        target_name = ['GU', 'SSp-un', 'MOp', 'VISC', 'SSs', 'PL', 'SSp-bfd', 'ACAd', 'RL', 'AL', 'DP', 'AUDpo', 'AM', 'MM', 'V1', 'LM', 'PM', 'P', 'RSPd']
        coords_target_name = target_name
        source_name = ['SSp-nm', 'SSp-lj', 'AIv', 'GU', 'AIp', 'FRP', 'SSp-ul', 'AId', 'SSp-un', 'SSp-ll', 'VISC', 'ORBl', 'PL', 'ORBm', 'ILA', 'MOp', 'MOs', 'SSs', 'ACAd', 'SSp-bfd', 'TEa', 'PERI', 'AUDv', 'AUDd', 'SSp-tr', 'ACAv', 'AUDp', 'ECT', 'A', 'RL', 'V1', 'RSPagl', 'TEp', 'RSPd', 'AM', 'RSPv', 'LM', 'MM', 'PM', 'AL', 'LI', 'POR', 'AUDpo', 'DP', 'LLA', 'PORa', 'P']
        accurate_name_dict = {}
        for name in all_name_coords:
            accurate_name_dict[name] = name
        name2namelist_dict = {}
        for name in source_name:
            name2namelist_dict[name] = [name]

    if animal_and_dataset_key == 'human':
        target_name = ['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', '4', '3b', 'FEF', 'PEF', '55b', 'V3A', 'RSC', 'POS2', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT', 'MT', 'A1', 'PSL', 'SFL', 'PCV', 'STV', '7Pm', '7m', 'POS1', '23d', 'v23ab', 'd23ab', '31pv', '5m', '5mv', '23c', '5L', '24dd', '24dv', '7AL', 'SCEF', '6ma', '7Am', '7PL', '7PC', 'LIPv', 'VIP', 'MIP', '1', '2', '3a', '6d', '6mp', '6v', 'p24pr', '33pr', 'a24pr', 'p32pr', 'a24', 'd32', '8BM', 'p32', '10r', '47m', '8Av', '8Ad', '9m', '8BL', '9p', '10d', '8C', '44', '45', '47l', 'a47r', '6r', 'IFJa', 'IFJp', 'IFSp', 'IFSa', 'p9-46v', '46', 'a9-46v', '9-46d', '9a', '10v', 'a10p', '10pp', '11l', '13l', 'OFC', '47s', 'LIPd', '6a', 'i6-8', 's6-8', '43', 'OP4', 'OP1', 'OP2-3', '52', 'RI', 'PFcm', 'PoI2', 'TA2', 'FOP4', 'MI', 'Pir', 'AVI', 'AAIC', 'FOP1', 'FOP3', 'FOP2', 'PFt', 'AIP', 'EC', 'PreS', 'H', 'ProS', 'PeEc', 'STGa', 'PBelt', 'A5', 'PHA1', 'PHA3', 'STSda', 'STSdp', 'STSvp', 'TGd', 'TE1a', 'TE1p', 'TE2a', 'TF', 'TE2p', 'PHT', 'PH', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP2', 'IP1', 'IP0', 'PFop', 'PF', 'PFm', 'PGi', 'PGs', 'V6A', 'VMV1', 'VMV3', 'PHA2', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', '31pd', '31a', 'VVC', '25', 's32', 'pOFC', 'PoI1', 'Ig', 'FOP5', 'p10p', 'p47r', 'TGv', 'MBelt', 'LBelt', 'A4', 'STSva', 'TE1m', 'PI', 'a32pr', 'p24']
        coords_target_name = ['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', '4', '3b', 'FEF', 'PEF', '55b', 'V3A', 'RSC--1', 'RSC--2', 'POS2', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT', 'MT', 'A1', 'PSL', 'SFL', 'PCV', 'STV', '7Pm', '7m', 'POS1', '23d--1', '23d--2', 'v23ab', 'd23ab', '31pv', '5m', '5mv--1', '5mv--2', '23c--1', '23c--2', '5L', '24dd', '24dv', '7AL', 'SCEF', '6ma', '7Am', '7PL', '7PC', 'LIPv', 'VIP', 'MIP', '1', '2', '3a', '6d', '6mp', '6v', 'p24pr', '33pr', 'a24pr', 'p32pr', 'a24', 'd32', '8BM', 'p32', '10r', '47m', '8Av', '8Ad', '9m', '8BL', '9p', '10d', '8C', '44', '45', '47l', 'a47r--1', 'a47r--2', '6r', 'IFJa', 'IFJp', 'IFSp', 'IFSa', 'p9-46v', '46', 'a9-46v', '9-46d', '9a', '10v', 'a10p', '10pp--1', '10pp--2', '11l', '13l', 'OFC--1', 'OFC--2', '47s', 'LIPd', '6a', 'i6-8', 's6-8', '43', 'OP4', 'OP1', 'OP2-3', '52', 'RI', 'PFcm', 'PoI2--1', 'PoI2--2', 'TA2', 'FOP4', 'MI', 'Pir--1', 'Pir--2', 'AVI', 'AAIC--1', 'AAIC--2', 'FOP1', 'FOP3', 'FOP2', 'PFt', 'AIP', 'EC', 'PreS--1', 'PreS--2', 'H', 'ProS--1', 'ProS--2', 'PeEc', 'STGa', 'PBelt', 'A5', 'PHA1', 'PHA3', 'STSda', 'STSdp', 'STSvp', 'TGd', 'TE1a', 'TE1p', 'TE2a', 'TF', 'TE2p', 'PHT', 'PH', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP2', 'IP1', 'IP0', 'PFop', 'PF', 'PFm', 'PGi', 'PGs', 'V6A', 'VMV1', 'VMV3', 'PHA2', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', '31pd', '31a', 'VVC', '25', 's32', 'pOFC--1', 'pOFC--2', 'PoI1--1', 'PoI1--2', 'Ig', 'FOP5', 'p10p', 'p47r--1', 'p47r--2', 'TGv', 'MBelt', 'LBelt', 'A4', 'STSva', 'TE1m', 'PI', 'a32pr', 'p24']
        source_name = ['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', '4', '3b', 'FEF', 'PEF', '55b', 'V3A', 'RSC', 'POS2', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT', 'MT', 'A1', 'PSL', 'SFL', 'PCV', 'STV', '7Pm', '7m', 'POS1', '23d', 'v23ab', 'd23ab', '31pv', '5m', '5mv', '23c', '5L', '24dd', '24dv', '7AL', 'SCEF', '6ma', '7Am', '7PL', '7PC', 'LIPv', 'VIP', 'MIP', '1', '2', '3a', '6d', '6mp', '6v', 'p24pr', '33pr', 'a24pr', 'p32pr', 'a24', 'd32', '8BM', 'p32', '10r', '47m', '8Av', '8Ad', '9m', '8BL', '9p', '10d', '8C', '44', '45', '47l', 'a47r', '6r', 'IFJa', 'IFJp', 'IFSp', 'IFSa', 'p9-46v', '46', 'a9-46v', '9-46d', '9a', '10v', 'a10p', '10pp', '11l', '13l', 'OFC', '47s', 'LIPd', '6a', 'i6-8', 's6-8', '43', 'OP4', 'OP1', 'OP2-3', '52', 'RI', 'PFcm', 'PoI2', 'TA2', 'FOP4', 'MI', 'Pir', 'AVI', 'AAIC', 'FOP1', 'FOP3', 'FOP2', 'PFt', 'AIP', 'EC', 'PreS', 'H', 'ProS', 'PeEc', 'STGa', 'PBelt', 'A5', 'PHA1', 'PHA3', 'STSda', 'STSdp', 'STSvp', 'TGd', 'TE1a', 'TE1p', 'TE2a', 'TF', 'TE2p', 'PHT', 'PH', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP2', 'IP1', 'IP0', 'PFop', 'PF', 'PFm', 'PGi', 'PGs', 'V6A', 'VMV1', 'VMV3', 'PHA2', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', '31pd', '31a', 'VVC', '25', 's32', 'pOFC', 'PoI1', 'Ig', 'FOP5', 'p10p', 'p47r', 'TGv', 'MBelt', 'LBelt', 'A4', 'STSva', 'TE1m', 'PI', 'a32pr', 'p24']
        accurate_name_dict = {}
        for name in all_name_coords:
            accurate_name_dict[name] = name
            if name in ['10pp--1', '10pp--2']:
                accurate_name_dict[name] = '10pp'
            if name in ['23c--1', '23c--2']:
                accurate_name_dict[name] = '23c'
            if name in ['23d--1', '23d--2']:
                accurate_name_dict[name] = '23d'
            if name in ['5mv--1', '5mv--2']:
                accurate_name_dict[name] = '5mv'
            if name in ['p47r--1', 'p47r--2']:
                accurate_name_dict[name] = 'p47r'
            if name in ['a47r--1', 'a47r--2']:
                accurate_name_dict[name] = 'a47r'
            if name in ['AAIC--1', 'AAIC--2']:
                accurate_name_dict[name] = 'AAIC'
            if name in ['Pir--1', 'Pir--2']:
                accurate_name_dict[name] = 'Pir'
            if name in ['OFC--1', 'OFC--2']:
                accurate_name_dict[name] = 'OFC'
            if name in ['pOFC--1', 'pOFC--2']:
                accurate_name_dict[name] = 'pOFC'
            if name in ['PoI1--1', 'PoI1--2']:
                accurate_name_dict[name] = 'PoI1'
            if name in ['PoI2--1', 'PoI2--2']:
                accurate_name_dict[name] = 'PoI2'
            if name in ['PreS--1', 'PreS--2']:
                accurate_name_dict[name] = 'PreS'
            if name in ['ProS--1', 'ProS--2']:
                accurate_name_dict[name] = 'ProS'
            if name in ['RSC--1', 'RSC--2']:
                accurate_name_dict[name] = 'RSC'
        name2namelist_dict = {}
        for name in source_name:
            name2namelist_dict[name] = [name]
            if name in ['10pp']:
                name2namelist_dict[name] = ['10pp--1', '10pp--2']
            if name in ['23c']:
                name2namelist_dict[name] = ['23c--1', '23c--2']
            if name in ['23d']:
                name2namelist_dict[name] = ['23d--1', '23d--2']
            if name in ['5mv']:
                name2namelist_dict[name] = ['5mv--1', '5mv--2']
            if name in ['a47r']:
                name2namelist_dict[name] = ['a47r--1', 'a47r--2']
            if name in ['p47r']:
                name2namelist_dict[name] = ['p47r--1', 'p47r--2']
            if name in ['AAIC']:
                name2namelist_dict[name] = ['AAIC--1', 'AAIC--2']
            if name in ['Pir']:
                name2namelist_dict[name] = ['Pir--1', 'Pir--2']
            if name in ['OFC']:
                name2namelist_dict[name] = ['OFC--1', 'OFC--2']
            if name in ['pOFC']:
                name2namelist_dict[name] = ['pOFC--1', 'pOFC--2']
            if name in ['PoI1']:
                name2namelist_dict[name] = ['PoI1--1', 'PoI1--2']
            if name in ['PoI2']:
                name2namelist_dict[name] = ['PoI2--1', 'PoI2--2']
            if name in ['PreS']:
                name2namelist_dict[name] = ['PreS--1', 'PreS--2']
            if name in ['ProS']:
                name2namelist_dict[name] = ['ProS--1', 'ProS--2']
            if name in ['RSC']:
                name2namelist_dict[name] = ['RSC--1', 'RSC--2']

    if len(source_name)>len(target_name):
        full_name = source_name.copy()
    else:
        full_name = target_name.copy()

    results = {'target_name': target_name, 'source_name': source_name, 'full_name': full_name, 'full_split_name': full_split_name, 'name_to_split_name_dict': name_to_split_name_dict, 'split_name_to_name_dict': split_name_to_name_dict}
    return results


def load_category_dict(animal, dataset_key, category_mode):
    '''
    读取区域的类别信息

    animal: str, 动物名称
    dataset_key: str, 数据集的名称
    category_mode: str, 类别的模式('raw', 'processed')
    '''
    category_dict = {}
    area_category = cf.load_df(r'../../data/'+animal+'_'+dataset_key+'/area_category.xlsx')
    for area in area_category.index:
        category_dict[str(area)] = area_category.loc[area, 'area_type']
    return category_dict
# endregion


# region load model parameters & results
def load_axon_source_target(outcomes_dir):
    '''
    读取轴突生长模型的source和target
    '''
    try:
        r = cf.load_dict_separate(outcomes_dir+'/axon_source_target/axon_source_target')
        cf.print_title('successfully load axon source and target')
        return r, True
    except:
        cf.print_title('failed to load axon source and target')
        return None, False


def load_SC_model_dict(outcomedir):
    '''
    读取SC模型的参数
    '''
    SC_model_dict = cf.load_dict_separate(outcomedir+'/SC_model_dict/SC_model_dict')
    return SC_model_dict
# endregion


# region axon_growth_model
def whether_in_area(p, polygon):
    '''
    Judge whether the point p in the area occupied by poly.
    '''
    point = shapely.geometry.Point(p[0], p[1])
    return polygon.intersects(point)


def pointpats_random_poisson(hull, intensity=None, size=None, seed=None):
    '''
    https://pysal.org/pointpats/_modules/pointpats/random.html#poisson

    but with random seed enabled
    '''
    if seed is None:
        seed = cf.get_time_seed()
    rng = cf.get_local_rng(seed=seed)

    if isinstance(hull, np.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = pointpats.geometry.prepare_hull(hull)
    n_observations, n_simulations, intensity = pointpats.random.parse_size_and_intensity(
        hull, intensity=intensity, size=size
    )

    result = np.empty((n_simulations, n_observations, 2))

    bbox = pointpats.geometry.bbox(hull)

    for i_replication in range(n_simulations):
        generating = True
        i_observation = 0
        while i_observation < n_observations:
            x, y = (
                rng.uniform(bbox[0], bbox[2]),
                rng.uniform(bbox[1], bbox[3]),
            )
            if pointpats.geometry.contains(hull, x, y):
                result[i_replication, i_observation] = (x, y)
                i_observation += 1
    return result.squeeze()


def generate_random_point_in_polygon(polygon, density, seed=None):
    '''
    在多边形内生成随机点

    polygon: shapely.geometry.Polygon, 多边形
    density: int, 点的密度
    '''
    return pointpats_random_poisson(polygon, size=int(density*polygon.area), seed=seed)


def get_neuron_num(neuron_density, polygon_dict, total_num, animal, dataset_key):
    '''
    获取每个区域的神经元数目

    neuron_density: dict, 每个区域的神经元密度
    polygon_dict: dict, 每个区域的多边形
    total_num: int, 总的神经元数目
    animal: str, 动物名称
    dataset_key: str, 数据集的名称
    
    return: dict, 每个区域的神经元数目(注意key是split的)
    '''
    split_neuron_density = broadcast_value_to_split_area(neuron_density, animal, dataset_key)
    neuron_num = {}
    for area in split_neuron_density.keys():
        neuron_num[area] = int(split_neuron_density[area]*polygon_dict[area].area)
    # 保证神经元数目的总和等于total_num
    neuron_num_sum = sum(neuron_num.values())
    for area in neuron_num.keys():
        neuron_num[area] = int(neuron_num[area]/neuron_num_sum*total_num)
    return neuron_num


def point_in_which_area(points, polygon_dict, whole_cortex_polygon):
    '''
    Get the area name for which the point belongs to.
    '''
    indices = []
    for s in points:
        if whether_in_area(s, whole_cortex_polygon):
            for name in polygon_dict.keys():
                if whether_in_area(s, polygon_dict[name]):
                    idx = name
                    break
        else:
            idx = 'out_of_areas'
        indices.append(idx)
    return indices


def get_direction_method_type(direction_method):
    '''
    获得方向的类型,方便简化模型选择时的代码
    '''
    if direction_method in ['song', 'song_without_self', 'song_exclude_inside']:
        return 'force'
    if direction_method in ['random']:
        return 'random'


def get_direction_func(direction_method):
    '''
    获取方向函数
    '''
    pass


def part_generate_axon_source_target(animal, dataset_key, direction_method, length_method, direction_params, length_params, na, point_density, part_index, model_type='send', use_part_index_as_seed=True, max_iteration=10):
    '''
    Generate the source and target of the axon growth model.

    point_density: dict, 密度,注意key不是split的
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    polygon_dict = animal_info.polygon_dict
    whole_cortex_polygon = animal_info.whole_cortex_polygon
    area_center_dict = animal_info.area_center_dict

    # 自动广播
    split_direction_params = {}
    split_length_params = {}
    for k, v in direction_params.items():
        if isinstance(v, dict):
            split_direction_params[k] = broadcast_value_to_split_area(v, animal, dataset_key)
        else:
            split_direction_params[k] = cf.create_dict(animal_info.full_split_name, default_value=v)
    for k, v in length_params.items():
        if isinstance(v, dict):
            split_length_params[k] = broadcast_value_to_split_area(v, animal, dataset_key)
        else:
            split_length_params[k] = cf.create_dict(animal_info.full_split_name, default_value=v)

    ss=[]
    tt=[]
    indices_ss=[]
    indices_tt=[]

    if use_part_index_as_seed:
        seed = part_index
    else:
        seed = cf.get_time_seed()
    rng = cf.get_local_rng(seed=seed)

    neuron_num = get_neuron_num(neuron_density=point_density, polygon_dict=polygon_dict, total_num=int(na), animal=animal, dataset_key=dataset_key)
    direction_method_type = get_direction_method_type(direction_method)

    for area, polygon in polygon_dict.items():
        for i in range(neuron_num[area]):
            s = pointpats.random.poisson(polygon, size=1)
            index_s = [area]
            index_t = ['out_of_areas'] # 默认值,确保进入循环
            try_num =0
            # to avoid endless iteration, if try_num is large, give up
            while 'out_of_areas' in index_t and try_num < max_iteration:
                try_num = try_num + 1
                if direction_method_type == 'force':
                    if direction_method == 'song':
                        forces = []
                        for force_area, force_center in area_center_dict.items():
                            forcei = (s - force_center) / (np.linalg.norm((s - force_center), ord=2) ** (split_direction_params['beta'][force_area] + 1))
                            forces.append(forcei)
                    
                    if direction_method == 'song_without_self':
                        forces = []
                        for force_area, force_center in area_center_dict.items():
                            if force_area != area:
                                forcei = (s - force_center) / (np.linalg.norm((s - force_center), ord=2) ** (split_direction_params['beta'][force_area] + 1))
                                forces.append(forcei)

                    if direction_method == 'song_exclude_inside':
                        forces = []
                        for force_area, force_center in area_center_dict.items():
                            if not whether_in_area(force_center, polygon):
                                forcei = (s - force_center) / (np.linalg.norm((s - force_center), ord=2) ** (split_direction_params['beta'][force_area] + 1))
                                forces.append(forcei)

                    # 处理send和receive
                    if model_type == 'send':
                        forces_sum = -np.sum(np.array(forces), axis=0)
                    else:
                        forces_sum = np.sum(np.array(forces), axis=0)
                    direction_growth = forces_sum / np.linalg.norm(forces_sum, ord=2) # direction of the axon
                else:
                    if direction_method == 'random':
                        angle = np.random.rand(1) * 2 * np.pi
                        direction_growth = np.array([np.cos(angle), np.sin(angle)])
                        direction_growth = direction_growth.reshape((2,))

                if length_method == 'exponential':
                    length = np.random.exponential(scale=split_length_params['Lambda'][area]) # length of the axon
                if length_method == 'power_law':
                    length = np.random.pareto(split_length_params['Lambda'][area])
                t = s + length * direction_growth # terminal position of the axon
                
                # check whether t is in the whole cortex
                index_t = point_in_which_area([t], polygon_dict, whole_cortex_polygon)

            if 'out_of_areas' not in index_t:
                tt.append(t)
                ss.append(s)
                indices_tt.append(index_t)
                indices_ss.append(index_s)


    if model_type=='receive':
        # just change ss and tt
        bb = ss
        ss = tt
        tt = bb
        indices_bb = indices_ss
        indices_ss = indices_tt
        indices_tt = indices_bb
    
    ss = np.array(ss)
    print(isinstance(ss, np.ndarray))
    tt = np.array(tt)
    indices_ss = np.array(indices_ss).flatten()
    indices_tt = np.array(indices_tt).flatten()
    results = {'ss': ss, 'tt': tt, 'indices_ss': indices_ss, 'indices_tt': indices_tt}
    return results


def generate_SC_from_axon_source_target(animal, dataset_key, outcomes_dir=None, indices_ss=None, indices_tt=None):
    '''
    Get the model SC and Data SC.

    note:
        CCM: connection count matrix
        CCMsym: symmetry connection count matrix
        CCMt: connection count matrix with self connection
        split_CCM: connection count matrix with split area name
    '''
    if indices_ss is None or indices_tt is None:
        axon_source_target, axon_source_target_exist = load_axon_source_target(outcomes_dir)
        indices_ss = axon_source_target['indices_ss']
        indices_tt = axon_source_target['indices_tt']
    
    animal_info = AnimalInfo(animal, dataset_key)
    full_name = animal_info.full_name
    full_split_name = animal_info.full_split_name

    # 将脑区名转换为序号
    idx_ss = area_name_to_index_array(indices_ss, full_split_name)
    idx_tt = area_name_to_index_array(indices_tt, full_split_name)

    # 生成connection_array
    split_CCMt_array = connection_pair_to_connection_matrix(idx_ss, idx_tt, area_num=len(full_split_name))

    # 转换为pd
    split_CCMt_model = pd.DataFrame(split_CCMt_array, columns=full_split_name, index=full_split_name)

    # 转换为normal
    CCMt_model = convert_split_df_to_normal(split_CCMt_model, animal, dataset_key, method='sum')
    
    # 去掉对角线
    CCM_model = cf.replace_df_diagonal(CCMt_model, replace_value=np.nan)

    # 生成对称的CCM
    CCMsym_model = pd.DataFrame((CCM_model.values+CCM_model.values.T)/2, index=CCM_model.index, columns=CCM_model.columns)

    # 整合
    SC_model_dict = {'CCM': CCM_model, 'CCMsym': CCMsym_model, 'CCMt': CCMt_model, 'split_CCMt': split_CCMt_model}

    # 生成normalized SC和log SC
    for k, v in SC_model_dict.items():
        SC_model_dict[k+'_normalized'] = normalize_SC(v, animal, dataset_key)
        SC_model_dict[k+'_normalized_log'] = log_SC(SC_model_dict[k+'_normalized'])

    cf.save_dict_separate(SC_model_dict, outcomes_dir+'/SC_model/SC_model_dict')
    return SC_model_dict


class AxonGrowthModel:
    def __init__(self, basedir, animal, dataset_key, direction_method, length_method, direction_params, length_params, na, model_type='send', use_neuron_density_to_generate=False, try_load_previous_results=True, process_num=1, use_part_index_as_seed=True, max_iteration=10):
        '''
        basedir: str, 基础的目录
        animal: str, 动物的名称
        dataset_key: str, 数据集的名称
        direction_method: str, 方向的方法
        length_method: str, 长度的方法
        direction_params: dict, 方向的参数
        length_params: dict, 长度的参数
        na: int, 神经元的数目
        use_density_to_generate: bool, 是否使用密度来生成神经元
        try_load_previous_results: bool, 是否尝试加载之前的运行结果
        '''
        self.basedir = basedir
        self.animal = animal
        self.dataset_key = dataset_key
        self.direction_method = direction_method
        self.length_method = length_method
        self.direction_params = direction_params
        self.length_params = length_params
        self.na = na
        self.model_type = model_type
        self.use_neuron_density_to_generate = use_neuron_density_to_generate
        self.try_load_previous_results = try_load_previous_results
        self.process_num = process_num
        self.use_part_index_as_seed = use_part_index_as_seed
        self.max_iteration = max_iteration

        self.value_dir_key = ['animal_and_dataset_key']
        self.search_params()

        animal_info = AnimalInfo(animal, dataset_key)

        if use_neuron_density_to_generate:
            self.point_density = animal_info.neuron_density.copy()
        else:
            self.point_density = cf.create_dict(animal_info.full_name, default_value=1)

    @property
    def outcomes_dir(self):
        '''
        保证outcomes_dir动态更新
        '''
        return cf.pj(self.timedir, 'outcomes')

    def search_params(self):
        '''
        寻找已有的使用相同参数的结果
        '''
        self.params = {'animal_and_dataset_key': cf.cat(self.animal, self.dataset_key), 'direction_method': self.direction_method, 'length_method': self.length_method, 'direction_params': self.direction_params, 'length_params': self.length_params, 'na': self.na, 'model_type': self.model_type, 'use_neuron_density_to_generate': self.use_neuron_density_to_generate}
        
        if self.try_load_previous_results:
            self.timedir, self.params_exist = cf.search_dict_subdir(dict_data=self.params, basedir=self.basedir, pkl_name='params', value_dir_key=self.value_dir_key, after_subdir='params')
        else:
            self.params_exist = False

    def save_params(self):
        '''
        保存参数
        '''
        self.timedir, _ = cf.save_timed_dir_dict(dict_data=self.params, basedir=self.basedir, dict_name='params', value_dir_key=self.value_dir_key, after_timedir='params')

    def run_model(self):
        '''
        运行模型
        '''
        if self.params_exist:
            # 如果params存在,则尝试读取axon
            self.axon_dict, self.axon_dict_exist = load_axon_source_target(self.outcomes_dir)
        else:
            # 如果params不存在,则重新生成
            self.axon_dict_exist = False

        if self.axon_dict_exist == False:
            # 运行axon_generation
            timer = cf.Timer(title='run axon growth model')
            timer.start()

            generation_kwargs = {'animal': self.animal, 'dataset_key': self.dataset_key, 'direction_method': self.direction_method, 'length_method': self.length_method, 'direction_params': self.direction_params, 'length_params': self.length_params, 'na': int(self.na/self.process_num), 'point_density': self.point_density, 'model_type': self.model_type, 'use_part_index_as_seed': self.use_part_index_as_seed, 'max_iteration': self.max_iteration}
            generate_for_list = list(range(self.process_num))
            axon_list = cf.multi_process_list_for(process_num=self.process_num, func=part_generate_axon_source_target, kwargs=generation_kwargs, for_list=generate_for_list, for_idx_name='part_index')
            # 合并结果
            self.axon_dict = cf.create_dict(keys=['ss', 'tt', 'indices_ss', 'indices_tt'], default_value=[])
            for r in axon_list:
                for k, v in r.items():
                    self.axon_dict[k].extend(v)
            # 将所有结果变成array
            self.axon_dict = {k: np.array(v) for k, v in self.axon_dict.items()}
            # 保存结果
            self.save_axon_dict()

            timer.stop()        

    def save_axon_dict(self):
        '''
        保存axon_dict
        '''
        # 此时保存参数,减少出现只有参数没有任何结果的文件夹
        self.save_params()
        # 保存结果
        cf.save_dict_separate(self.axon_dict, self.outcomes_dir+'/axon_source_target/axon_source_target')

    @property
    def SC_model_dict(self):
        '''
        获取SC_model_dict
        '''
        try:
            _SC_model_dict, SC_model_dict_exist = load_SC_model_dict(self.outcomes_dir)
            return _SC_model_dict
        except:
            _SC_model_dict = generate_SC_from_axon_source_target(self.animal, self.dataset_key, outcomes_dir=self.outcomes_dir, indices_ss=self.axon_dict['indices_ss'], indices_tt=self.axon_dict['indices_tt'])
            return _SC_model_dict

    def RMAE(self):
        pass

    def useless_code():
        pass

    @cf.to_be_improved
    def force_vector(self, points, beta, centers):
        """
        For each point (acctually s), using force_direction to get the target(t). The length between t and s should be 1.
        notation: s: initial positions of the axon
            t: terminal positions of the axon
            centers: positions of area centers
        """
        tt=[]
        if self.direction=='song':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    Ri = centers[center_index]
                    s = np.array(s)
                    forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                    forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='song_without_self':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    if center_index!=self.polygon_index([s])[0]:
                        Ri = centers[center_index]
                        s = np.array(s)
                        forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                        forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='song_exclude_self':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    if not self.whether_in_area(centers[center_index], self.pgon[self.polygon_index([s])[0]]):
                        Ri = centers[center_index]
                        s = np.array(s)
                        forcei = (s-Ri)/(np.linalg.norm((s-Ri),ord=2)**(beta[center_index]+1))
                        forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='exponential':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    Ri = centers[center_index]
                    s = np.array(s)
                    forcei = (s-Ri)/np.exp(np.linalg.norm((s-Ri),ord=2)*(beta[center_index]))
                    forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        if self.direction=='exponential_without_self':
            for s in points:
                forces=[]
                # for center_index in self.all_name_coords:
                for center_index in centers.keys():
                    if center_index!=self.polygon_index([s])[0]:
                        Ri = centers[center_index]
                        s = np.array(s)
                        forcei = (s-Ri)/np.exp(np.linalg.norm((s-Ri),ord=2)*(beta[center_index]))
                        forces.append(forcei)
                if self.model_type=='send':
                    forces_sum = -np.sum(np.array(forces),axis=0)
                else:
                    forces_sum = np.sum(np.array(forces),axis=0)
                direction_growth = forces_sum/np.linalg.norm(forces_sum,ord=2) # direction of the axon
                t = s+direction_growth # terminal position of the axon
                tt.append(t)
        return tt




    # def part_visualization(self, FLN_list, FLN_titles, file_name, plot_names, sources, targets, plot_type, draw_axon_arrow):
        
    #     print('donot use it')
    #     print(1/0)
        
    #     '''Draw several plots, green means the target location, red means source location. More red, larger FLN. FLN should be a list, for example, FLN_list = [FLN_data, FLN_model], FLN_titles should be the corresponding titles, for example, FLN_title = ['FLN_data', 'FLN_model']. Due to the incompleteness of data, we mask missing areas by black(this is only done when plot_type is source, because for plot_type=target, the data is complete.)'''
    #     color_dict = {'target': 'g', 'source': 'r'}
    #     target_source_transform = {'target': 'source', 'source': 'target'}
    #     arrow_size_dict = {'marmoset':0.25, 'macaque40':0.4, 'macaque29':0.4}
    #     for plot_name in plot_names:
    #         plot_name_list = self.name2namelist_dict[plot_name]
    #         figname = '_'+plot_type+'='+str(plot_name)+'.png'
    #         if plot_type=='source':
    #             figure ,axes = plt.subplots(1, len(FLN_list), figsize=(len(FLN_list)*10, 10))
    #             for iii in range(len(FLN_list)):
    #                 max_area = ''
    #                 FLN = FLN_list[iii]
    #                 axes[iii].plot(0, 0, label='target:green/source:red')
    #                 axes[iii].legend(fontsize=15)
    #                 for name in self.all_name_coords:
    #                     accurate_name = self.accurate_name_dict[name]
    #                     if plot_type=='source':
    #                         try:
    #                             alpha = FLN.loc[plot_name, accurate_name]/np.max(FLN.loc[plot_name, self.target_name])
    #                         except:
    #                             alpha = 0
    #                         if str(alpha)=='nan':
    #                             alpha = 0
    #                         elif np.allclose(alpha, 1):
    #                             max_area = accurate_name
    #                         if accurate_name not in self.target_name:
    #                             if accurate_name!=plot_name:
    #                                 axes[iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c='black', zorder=1)
    #                         else:
    #                             axes[iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], alpha=alpha, c=color_dict[target_source_transform[plot_type]], zorder=1)
    #                     boundary = self.area_coords[name]
    #                     boundary = np.append(boundary, [boundary[0]], axis=0)
    #                     axes[iii].plot(boundary.T[0], boundary.T[1], c='black', linewidth=1, zorder=1)
    #                     axes[iii].text(self.name_location[name][0], self.name_location[name][1], name, fontsize=7, font={'style':'normal','weight':'black'})
    #                 for name in plot_name_list:
    #                     axes[iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c=color_dict[plot_type], zorder=1)
    #                 if (self.direction=='song') or (self.direction=='exponential') or (self.direction=='song_without_self') or (self.direction=='exponential_without_self') or (self.direction=='song_exclude_self'):
    #                     # draw arrows
    #                     for i in range(len(sources)):
    #                         axes[iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/2, (targets[i][1]-sources[i][1])/2, fc='b', ec='b', head_width=arrow_size_dict[self.animal]/2, head_length=arrow_size_dict[self.animal]/2, zorder=2)
    #                         # if plot_type=='source':
    #                         #     for name in plot_name_list:
    #                         #         if self.whether_in_area([sources[i][0], sources[i][1]], self.area_coords[name]):
    #                         #             axes[iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/4, (targets[i][1]-sources[i][1])/4, fc='b', ec='b', head_width=0.15, head_length=0.15)
    #                         # else:
    #                         #     axes[iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/2, (targets[i][1]-sources[i][1])/2, fc='b', ec='b', head_width=0.2, head_length=0.2)
    #                 if plot_type=='source':
    #                     axes[iii].set_title(FLN_titles[iii]+'\nmax fln in this plot: area '+max_area+'\n'+str(np.max(FLN.loc[plot_name, self.target_name])), fontdict={'weight': 'normal', 'size': 25})
    #                     if FLN_titles[iii]=='FLN_model' and draw_axon_arrow==True:
    #                         coords_plot_names = self.name2namelist_dict[plot_name]
    #                         for coords_plot_name in coords_plot_names:
    #                             index = np.where(((self.indices_ss.flatten()==coords_plot_name)*(self.indices_tt!=coords_plot_name))==True)
    #                             ss_partial = np.array(self.ss[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
    #                             tt_partial = np.array(self.tt[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
    #                             for i in range(len(ss_partial)):
    #                                 axes[iii].arrow(ss_partial[i][0], ss_partial[i][1], tt_partial[i][0]-ss_partial[i][0], tt_partial[i][1]-ss_partial[i][1], fc='orange', ec='orange', head_width=arrow_size_dict[self.animal], head_length=arrow_size_dict[self.animal], zorder=3)
    #             plt.savefig(str(file_name)+figname, dpi=300)
    #             plt.close()
    #         else:
    #             figure ,axes = plt.subplots(2, len(FLN_list), figsize=(len(FLN_list)*10, 2 * 10))
    #             for mask_index in range(2):
    #                 for iii in range(len(FLN_list)):
    #                     max_area = ''
    #                     FLN = FLN_list[iii]
    #                     axes[mask_index, iii].plot(0, 0, label='target:green/source:red')
    #                     axes[mask_index, iii].legend(fontsize=15)
    #                     for name in self.all_name_coords:
    #                         accurate_name = self.accurate_name_dict[name]
    #                         if plot_type=='target':
    #                             alpha = FLN.loc[accurate_name, plot_name]/np.max(FLN.loc[:, plot_name])
    #                             if str(alpha)=='nan':
    #                                 alpha = 0
    #                             elif np.allclose(alpha, 1):
    #                                 max_area = accurate_name
    #                             axes[mask_index, iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], alpha=alpha, c=color_dict[target_source_transform[plot_type]], zorder=1)
    #                         boundary = self.area_coords[name]
    #                         boundary = np.append(boundary, [boundary[0]], axis=0)
    #                         axes[mask_index, iii].plot(boundary.T[0], boundary.T[1], c='black', linewidth=1, zorder=1)
    #                         axes[mask_index, iii].text(self.name_location[name][0], self.name_location[name][1], name, fontsize=7, font={'style':'normal','weight':'black'})
    #                     for name in plot_name_list:
    #                         axes[mask_index, iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c=color_dict[plot_type], zorder=1)
    #                     if (self.direction=='song') or (self.direction=='exponential') or (self.direction=='song_without_self') or (self.direction=='exponential_without_self') or (self.direction=='song_exclude_self'):
    #                         # draw arrows
    #                         for i in range(len(sources)):
    #                             axes[mask_index, iii].arrow(sources[i][0], sources[i][1], (targets[i][0]-sources[i][0])/2, (targets[i][1]-sources[i][1])/2, fc='b', ec='b', head_width=arrow_size_dict[self.animal]/2, head_length=arrow_size_dict[self.animal]/2, zorder=2)
    #                     if plot_type=='target':
    #                         if mask_index==0:
    #                             axes[mask_index, iii].set_title(FLN_titles[iii]+'\nmax fln in this plot: area '+max_area+'\n'+str(np.max(FLN.loc[:, plot_name])), fontdict={'weight': 'normal', 'size': 25})
    #                         if FLN_titles[iii]=='FLN_model' and draw_axon_arrow==True:
    #                             coords_plot_names = self.name2namelist_dict[plot_name]
    #                             for coords_plot_name in coords_plot_names:
    #                                 index = np.where(((self.indices_tt.flatten()==coords_plot_name)*(self.indices_ss!=coords_plot_name))==True)
    #                                 random.shuffle(index)
    #                                 ss_partial = np.array(self.ss[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
    #                                 tt_partial = np.array(self.tt[index])[0:int(np.min([int(30/len(coords_plot_names)), len(index[0])]))]
    #                                 for i in range(len(ss_partial)):
    #                                     axes[mask_index, iii].arrow(ss_partial[i][0], ss_partial[i][1], tt_partial[i][0]-ss_partial[i][0], tt_partial[i][1]-ss_partial[i][1], fc='orange', ec='orange', head_width=arrow_size_dict[self.animal], head_length=arrow_size_dict[self.animal], zorder=3)
    #                     if mask_index==1:
    #                         for name in self.all_name_coords:
    #                             accurate_name = self.accurate_name_dict[name]
    #                             if (accurate_name not in self.target_name) and (accurate_name!=plot_name):
    #                                 axes[mask_index, iii].fill(self.area_coords[name].T[0], self.area_coords[name].T[1], c='black', zorder=1)
    #             plt.savefig(str(file_name)+figname, dpi=300)
    #             plt.close()


    # def simultaneously_visualization(self, FLN_list, FLN_titles, file_name, draw_axon_arrow, multiprocess):
        
    #     print('donot use it')
    #     print(1/0)
        
    #     '''Using part_visualization to generate FLN visualization plots, accelerated by multiprocess.'''
    #     if (self.direction=='song') or (self.direction=='exponential') or (self.direction=='song_without_self') or (self.direction=='exponential_without_self') or (self.direction=='song_exclude_self'):
    #         # generate the vector field
    #         x_list = np.linspace(self.x_min, self.x_max, self.mesh_dict[self.animal])
    #         y_list = np.linspace(self.y_min, self.y_max, self.mesh_dict[self.animal])
    #         sources = []
    #         for x in x_list:
    #             for y in y_list:
    #                 if self.whether_in_area([x, y], self.pgon_whole['whole_cortex'])==True and self.polygon_index([[x, y]])[0]!='out_of_areas':
    #                     sources.append([x, y])
    #         targets = self.force_vector(sources,self.beta,self.force_centers_dict)
    #         sources = np.array(sources)
    #         targets = np.array(targets)
    #     else:
    #         sources = np.nan
    #         targets = np.nan
    #     # for each target and source, draw a plot
    #     if multiprocess:
    #         process_list = []
    #         num = 8
    #         for part_index in range(num):
    #             targetname = self.target_name[ceil(len(self.target_name)/num)*part_index:ceil(len(self.target_name)/num)*part_index+ceil(len(self.target_name)/num)]
    #             p = Process(target=self.part_visualization, args=(FLN_list, FLN_titles, file_name, targetname, sources, targets, 'target', draw_axon_arrow))
    #             p.start()
    #             process_list.append(p)
    #         for part_index in range(num):
    #             sourcename = self.source_name[ceil(len(self.source_name)/num)*part_index:ceil(len(self.source_name)/num)*part_index+ceil(len(self.source_name)/num)]
    #             p = Process(target=self.part_visualization, args=(FLN_list, FLN_titles, file_name, sourcename, sources, targets, 'source', draw_axon_arrow))
    #             p.start()
    #             process_list.append(p)
    #         for i in process_list:
    #             p.join()
    #     else:
    #         self.part_visualization(FLN_list, FLN_titles, file_name, self.target_name, sources, targets, 'target', draw_axon_arrow)
    #         self.part_visualization(FLN_list, FLN_titles, file_name, self.source_name, sources, targets, 'source', draw_axon_arrow)


    # def do_visualization(self, FLN_data, data_type, draw_axon_arrow=False, multiprocess=True):
    #     '''Using simultaneously_visualization to get FLN visualization.'''
        
    #     print('donot use it')
    #     print(1/0)

    #     print('----------------FLN visualization----------------')
    #     mkdir(self.basedir+'/FLN_weight_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow))
    #     self.simultaneously_visualization([FLN_data, self.FLN_model], ['FLN_data_'+str(len(self.target_name)), 'FLN_model'], file_name=self.basedir+'/FLN_weight_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow)+'/FLN', draw_axon_arrow=draw_axon_arrow, multiprocess=multiprocess)

    #     FLN_data_topo = pd.DataFrame(np.zeros(np.shape(FLN_data.values)), index=FLN_data.index, columns=FLN_data.columns)
    #     FLN_data_topo.values[np.nonzero(FLN_data.values)] = 1
    #     FLN_model_topo = pd.DataFrame(np.zeros(np.shape(self.FLN_model.values)), index=self.FLN_model.index, columns=self.FLN_model.columns)
    #     FLN_model_topo.values[np.nonzero(self.FLN_model.values)] = 1
    #     mkdir(self.basedir+'/FLN_topo_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow))
    #     self.simultaneously_visualization([FLN_data_topo, FLN_model_topo], ['FLN_data_'+str(len(self.target_name)), 'FLN_model'], file_name=self.basedir+'/FLN_topo_visualization_'+data_type+'_draw_arrow='+str(draw_axon_arrow)+'/FLN', draw_axon_arrow=draw_axon_arrow, multiprocess=multiprocess)


    # def draw_correlation(self, target_list, data_FLN_square, model_FLN_square, data_FLN_rectangle, model_FLN_rectangle, filename):
    #     print('dont use')
    #     print(1/0)
    #     '''Draw correlation plot for each target in target_list. Note that nan is possible because if the data contains all 0, the variance is zero.'''
    #     correlation_list_square = []
    #     correlation_list_rectangle = []
    #     for target in target_list:
    #         correlation_list_square.append(np.corrcoef(data_FLN_square.loc[:, target], model_FLN_square.loc[:, target])[0, 1])
    #         correlation_list_rectangle.append(np.corrcoef(data_FLN_rectangle.loc[:, target], model_FLN_rectangle.loc[:, target])[0, 1])
    #     flatten_corr_square = np.corrcoef(data_FLN_square.loc[:, target_list].values.flatten(), model_FLN_square.loc[:, target_list].values.flatten())[0, 1]
    #     flatten_corr_rectangle = np.corrcoef(data_FLN_rectangle.loc[:, target_list].values.flatten(), model_FLN_rectangle.loc[:, target_list].values.flatten())[0, 1]
    #     plt.figure(figsize=(12,7))
    #     plt.plot(range(len(target_list)), correlation_list_square, 'o-', label='square: mean_corr='+str(round(np.mean(correlation_list_square), 3))+'  flatten_corr='+str(round(flatten_corr_square, 3)))
    #     plt.plot(range(len(target_list)), correlation_list_rectangle, 'o-', label='rectangle: mean_corr='+str(round(np.mean(correlation_list_rectangle), 3))+'  flatten_corr='+str(round(flatten_corr_rectangle, 3)))
    #     plt.xticks(range(len(target_list)), target_list, rotation=50, fontsize=8)
    #     plt.xlabel('target areas')
    #     plt.ylabel('correlation')
    #     plt.ylim([-0.2, 1.1])
    #     plt.legend(loc='upper right')
    #     plt.savefig(filename+'.png', dpi=300)
    #     plt.close()

        
    #     # plt.figure(figsize=(12,7))
    #     # plt.plot(range(len(target_list)), correlation_list_square, 'o-', label='square: mean_corr='+str(round(np.mean(correlation_list_square), 3))+'  flatten_corr='+str(round(flatten_corr_square, 3)))
    #     # plt.xticks(range(len(target_list)), target_list, rotation=50, fontsize=8)
    #     # plt.xlabel('target areas')
    #     # plt.ylabel('correlation')
    #     # plt.ylim([-0.2, 1.1])
    #     # plt.legend()
    #     # plt.savefig(filename+'_square.png', dpi=300)
    #     # plt.close()


    #     # sort_id = np.argsort(correlation_list_square)
    #     # plt.figure(figsize=(12,7))
    #     # plt.plot(range(len(target_list)), np.array(correlation_list_square)[sort_id], 'o-', label='square: mean_corr='+str(round(np.mean(correlation_list_square), 3))+'  flatten_corr='+str(round(flatten_corr_square, 3)))
    #     # plt.xticks(range(len(target_list)), np.array(target_list)[sort_id], rotation=50, fontsize=8)
    #     # plt.xlabel('target areas')
    #     # plt.ylabel('correlation')
    #     # plt.ylim([-0.2, 1.1])
    #     # plt.legend()
    #     # plt.savefig(filename+'_square_sorted.png', dpi=300)
    #     # plt.close()


    #     fig, ax = plt.subplots(figsize=(12,7))
    #     ax.bar(range(len(target_list)), correlation_list_rectangle, label='rectangle: mean_corr='+str(round(np.mean(correlation_list_rectangle), 3))+'  flatten_corr='+str(round(flatten_corr_rectangle, 3)))
    #     ax.set_xticks(range(len(target_list)), target_list, rotation=90, fontsize=14)
    #     ax.set_xlabel('target areas', fontsize=20)
    #     ax.set_ylabel('correlation', fontsize=20)
    #     ax.tick_params(axis='y', labelsize=16)
    #     ax.set_ylim([-0.2, 1.1])
    #     ax.legend(fontsize=16, loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig(filename+'_rectangle.png', dpi=300)
    #     plt.close()


    #     sort_id = np.argsort(correlation_list_rectangle)
    #     fig, ax = plt.subplots(figsize=(12,7))
    #     ax.bar(range(len(target_list)), np.array(correlation_list_rectangle)[sort_id], label='rectangle: mean_corr='+str(round(np.mean(correlation_list_rectangle), 3))+'  flatten_corr='+str(round(flatten_corr_rectangle, 3)))
    #     ax.set_xticks(range(len(target_list)), np.array(target_list)[sort_id], rotation=90, fontsize=14)
    #     ax.set_xlabel('target areas', fontsize=20)
    #     ax.set_ylabel('correlation', fontsize=20)
    #     ax.tick_params(axis='y', labelsize=16)
    #     ax.set_ylim([-0.2, 1.1])
    #     ax.legend(fontsize=16, loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig(filename+'_rectangle_sorted.png', dpi=300)
    #     plt.close()
    #     return correlation_list_square, correlation_list_rectangle, flatten_corr_square, flatten_corr_rectangle


    # def do_draw_correlation(self, plot_target_name, FLN_data, FLN_model, plot_name):
    #     print('dont use')
    #     print(1/0)
    #     '''Using draw_correlation to draw correlation plots and get the corr matrix and list.'''
    #     results = {}
    #     FLN_data_RECTANGLE = FLN_data[plot_target_name]
    #     FLN_data_SQUARE = FLN_data.loc[plot_target_name, plot_target_name]
    #     FLN_model_RECTANGLE = FLN_model[plot_target_name]
    #     FLN_model_SQUARE = FLN_model.loc[plot_target_name, plot_target_name]
    #     results['SQUARE_weight'], results['RECTANGLE_weight'], results['flatten_SQUARE_weight'], results['flatten_RECTANGLE_weight']  = self.draw_correlation(plot_target_name, FLN_data_SQUARE, FLN_model_SQUARE, FLN_data_RECTANGLE, FLN_model_RECTANGLE, self.basedir+'/correlation_weight_'+str(plot_name))

    #     FLN_data_topo = pd.DataFrame(np.zeros(np.shape(FLN_data.values)), index=FLN_data.index, columns=FLN_data.columns)
    #     FLN_data_topo.values[np.nonzero(FLN_data.values)] = 1
    #     FLN_data_topo_RECTANGLE = FLN_data_topo[plot_target_name]
    #     FLN_data_topo_SQUARE = FLN_data_topo.loc[plot_target_name, plot_target_name]
    #     FLN_model_topo = pd.DataFrame(np.zeros(np.shape(FLN_model.values)), index=FLN_model.index, columns=FLN_model.columns)
    #     FLN_model_topo.values[np.nonzero(FLN_model.values)] = 1
    #     FLN_model_topo_RECTANGLE = FLN_model_topo[plot_target_name]
    #     FLN_model_topo_SQUARE = FLN_model_topo.loc[plot_target_name, plot_target_name]
    #     results['SQUARE_topo'], results['RECTANGLE_topo'], results['flatten_SQUARE_topo'], results['flatten_RECTANGLE_topo']  = self.draw_correlation(plot_target_name, FLN_data_topo_SQUARE, FLN_model_topo_SQUARE, FLN_data_topo_RECTANGLE, FLN_model_topo_RECTANGLE, self.basedir+'/correlation_topo_'+str(plot_name))
    #     with open(self.basedir+'/results.txt', 'w') as f:
    #         f.write(plot_name+'\r\n')
    #         for key in results.keys():
    #             f.write(key+str(np.mean(results[key]))+'\r\n')
    #     return results


    # def target_location_visualization(self):
    #     '''Plot the target location (where the injections belong to).'''
    #     mkdir('../../data/'+str(self.animal)+'/target_location_visualization/')
    #     file_name = r'../../data/'+str(self.animal)+'/target_location_visualization/target_location_visualization_'+str(len(self.target_name))+'.jpg'
    #     targets = self.coords_target_name
    #     fig, ax = plt.subplots(figsize=(20, 20))
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['left'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.axis('off')
    #     for name in self.all_name_coords:
    #         boundary = self.area_coords[name]
    #         boundary = np.append(boundary, [boundary[0]], axis=0)
    #         plt.plot(boundary.T[0], boundary.T[1], c='black')
    #         plt.text(self.name_location[name][0], self.name_location[name][1], name, fontsize=20)
    #     for target in targets:
    #         plt.fill(self.area_coords[target].T[0], self.area_coords[target].T[1], c='g')
    #     plt.savefig(str(file_name), bbox_inches='tight', pad_inches=0.0, dpi=300)
    #     plt.close()

    
    # def do_visualize_connection(self):
    #     if self.rearranged==False:
    #         visualize_connection(self.ss, self.indices_ss, self.indices_tt, self.rearranged, self.basedir, self.area_coords, self.all_name_coords, self.accurate_name_dict, self.name_location)
    #         visualize_connection_with_injection(self.ss, self.tt, self.indices_ss, self.indices_tt, self.rearranged, self.basedir, self.area_coords, self.all_name_coords, self.accurate_name_dict, self.name_location, self.animal)
    #     else:
    #         visualize_connection(self.new_ss, self.new_indices_ss, self.new_indices_tt, self.rearranged, self.basedir, self.area_coords, self.all_name_coords, self.accurate_name_dict, self.name_location)


    # def degree_hier_plot(self):
    #     if self.hierarchy_exist:
    #         mkdir(self.basedir+'/degree_hier_relation/')
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #         for area in self.hierarchy.keys():
    #             ax.scatter(self.hierarchy[area], np.count_nonzero(self.FLN_model.loc[:, area]), color=default_blue)
    #         ax.set_xlabel('hierarchy')
    #         ax.set_ylabel('area_level_degree')
    #         plt.savefig(self.basedir+'/degree_hier_relation/degree_hier_plot.png', dpi=300)
    #         plt.close()


    # def self_connection_plot(self):
    #     if self.animal=='marmoset':
    #         clustered_source_name, clustered_target_name = clustered_name(self.animal)
    #         intrinsic_proportion(self.animal, self.basedir, clustered_target_name, self.connection_count_matrix_with_self_connection)
    #     if self.animal=='marmoset-V4':
    #         clustered_source_name, clustered_target_name = clustered_name(self.animal)
    #         FLNt_data = pd.read_excel(r'../../data/'+str(self.animal)+'/FLNt_data.xlsx', index_col=0)
    #         FLNt_data.index = FLNt_data.index.astype(str)
    #         FLNt_data.columns = FLNt_data.columns.astype(str)

    #         intrinsic_proportion_data = []
    #         for target in clustered_target_name:
    #             intrinsic_proportion_data.append(FLNt_data.loc[target, target])

    #         FLNt_model = self.connection_count_matrix_with_self_connection.copy()
    #         FLNt_model = FLNt_model / FLNt_model.sum(axis=0)
    #         intrinsic_proportion_model = []
    #         for target in clustered_target_name:
    #             intrinsic_proportion_model.append(FLNt_model.loc[target, target])
    #         uf.mkdir(self.basedir+'/self_connection_proportion/')
    #         uf.intrisic_bar_scatter_plot(clustered_target_name, self.basedir, intrinsic_proportion_data, intrinsic_proportion_model, corrected_str='')

    
    # def do_rearrange(self):
    #     print('donot use it')
    #     print(1/0)
    #     self.new_ss, self.new_tt, self.new_indices_ss, self.new_indices_tt = rearrange(self.ss, self.tt, self.indices_ss, self.indices_tt, self.animal)
    #     self.rearranged = True
    #     np.save(self.basedir+'/full_data/new_ss_'+str(self.na)+'.npy', self.new_ss)
    #     np.save(self.basedir+'/full_data/new_tt_'+str(self.na)+'.npy', self.new_tt)
    #     np.save(self.basedir+'/full_data/new_indices_ss_'+str(self.na)+'.npy', self.new_indices_ss)
    #     np.save(self.basedir+'/full_data/new_indices_tt_'+str(self.na)+'.npy', self.new_indices_tt)

    
    # def draw_area_size_degree_plot(self):
    #     area_size_degree_plot(self.FLN_data, self.FLN_model, self.animal, self.basedir+'/area_size_and_degree/')

    
    # def do_FLN_visualization(self):
    #     if self.animal in ['mouse', 'marmoset', 'macaque', 'marmoset-V4']:
    #         visualize_fln(self.FLN_model, self.FLN_data, self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='FLN', model_type='FLN', vmax=0.25)
    #     if self.animal=='marmoset':
    #         visualize_fln(self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='dMRI', model_type='dMRI', vmax=0.25)
    #     if self.animal=='human':
    #         visualize_fln(self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='dMRI', model_type='dMRI', vmax=0.25)
    #         # visualize_fln(self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.DTI_data/np.sum(self.DTI_data.values)*20, self.target_name, self.source_name, self.area_coords, self.name_location, self.name2namelist_dict, self.basedir, data_type='dMRI', model_type='dMRI', vmax=0.25)


    # def do_remove_long_range_area_by_area_and_draw_corr(self):
    #     print('need change, dont use')
    #     print(1/0)
    #     distance_matrix_2D = pd.read_excel('../../data/'+str(self.animal)+'/distance_matrix_2D.xlsx', index_col=0)
    #     remove_long_range_area_by_area_and_draw_corr(self.FLN_model, self.FLN_data, distance_matrix_2D, self.target_name, np.linspace(0, np.max(distance_matrix_2D.values), 30, endpoint=True)[1:], self.basedir)


    # def do_remove_long_range_and_cal_corr(self):
    #     if self.animal!='human':
    #         remove_long_range_and_cal_corr(self.animal, self.FLN_data, self.FLN_model, self.target_name, self.distance_matrix_2D, self.basedir)
    #     else:
    #         remove_long_range_and_cal_corr(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.distance_matrix_2D, self.basedir)
    #         # remove_long_range_and_cal_corr(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, self.distance_matrix_2D, self.basedir)


    # def do_draw_angle_distance_kde_plot(self):
    #     if self.animal!='human':
    #         do_draw_kde_plot(self.animal, self.FLN_data, self.FLN_model, self.basedir)
    #     else:
    #         do_draw_kde_plot(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir)
    #         # do_draw_kde_plot(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.basedir)


    # def do_compare_area_by_area(self):
    #     optimal_para = pd.read_excel('../../results/optimal/optimal_para.xlsx', index_col=0)
    #     if self.animal=='marmoset' or self.animal=='marmoset-V4':
    #         threshold = 5 * optimal_para.loc['marmoset', 'Lambda']
    #     else:
    #         threshold = 5 * optimal_para.loc[self.animal, 'Lambda']
    #     if self.animal!='human':
    #         compare_area_by_area(self.FLN_data, self.FLN_model, self.target_name, 'extrinsic', self.basedir, self.animal, 'remove_diagonal', 'FLNe')
    #         if self.animal!='mouse' and self.animal!='marmoset-V4':
    #             FLN_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.FLN_data, threshold, self.animal, distance_type='2D')
    #             compare_area_by_area(FLN_data_without_prefrontal_long_range, self.FLN_model, self.target_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'FLNe')
    #         if self.animal=='marmoset':
    #             FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
    #             FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
    #             FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
    #             compare_area_by_area(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name, 'with_intrinsic', self.basedir, self.animal, 'dont_remove_diagonal', 'FLNt')

    #             FLN_data_without_prefrontal_long_range = remove_prefrontal_long_range(FLN_with_intrinsic_data, threshold, self.animal, distance_type='2D')
    #             compare_area_by_area(FLN_data_without_prefrontal_long_range, FLN_with_intrinsic_model, self.target_name, 'with_intrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'FLNt')


    #             FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'connection_density')
    #             compare_area_by_area(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, 'with_corrected_by_connection_density_intrinsic', self.basedir, self.animal, 'dont_remove_diagonal', 'FLNt')

    #             FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'neuron_density')
    #             compare_area_by_area(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, 'with_corrected_by_neuron_density_intrinsic', self.basedir, self.animal, 'dont_remove_diagonal', 'FLNt')


    #             compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.source_name, 'DTI_extrinsic', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
    #             # compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.source_name, 'DTI_extrinsic', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
                
    #             DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
    #             compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.source_name, 'DTI_extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
    #             DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
    #             # compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.source_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'DTIn')
    #     else:
    #         compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, 'extrinsic', self.basedir, self.animal, 'remove_diagonal', 'dMRI')
    #         # compare_area_by_area(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, 'extrinsic', self.basedir, self.animal, 'remove_diagonal', 'dMRI')
            
    #         DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
    #         compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'dMRI')
    #         DTI_data_without_prefrontal_long_range = remove_prefrontal_long_range(self.DTI_data, threshold, self.animal, distance_type='2D')
    #         # compare_area_by_area(DTI_data_without_prefrontal_long_range/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, 'extrinsic_without_prefrontal_long_range', self.basedir, self.animal, 'remove_diagonal', 'dMRI')


    # def do_corr_skewness_plot(self):
    #     if self.animal=='marmoset':
    #         FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
    #         FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
    #         FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
    #         corr = self.do_area_by_area_correlation()
    #         corr_with_intrinsic = area_by_area_correlation(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name)
    #         corr_list = []
    #         corr_with_intrinsic_list = []
    #         for target in self.target_name:
    #             corr_list.append(corr[target])
    #             corr_with_intrinsic_list.append(corr_with_intrinsic[target])
    #         corr_skewness_plot(corr_list, corr_with_intrinsic_list, self.animal, self.basedir)


    # def do_corr_direction_preference(self):
    #     if self.animal=='marmoset':
    #         FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
    #         FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
    #         FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
    #         corr = self.do_area_by_area_correlation()
    #         corr_with_intrinsic = area_by_area_correlation(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name)
    #         corr_list = []
    #         corr_with_intrinsic_list = []
    #         for target in self.target_name:
    #             corr_list.append(corr[target])
    #             corr_with_intrinsic_list.append(corr_with_intrinsic[target])
    #         corr_direction_preference_plot(self.animal, corr_list, corr_with_intrinsic_list, self.target_name, self.basedir)
    #     elif self.animal=='human':
    #         corr = self.do_area_by_area_correlation()
    #         corr_list = []
    #         for target in self.target_name:
    #             corr_list.append(corr[target])
    #         corr_direction_preference_plot(self.animal, corr_list, np.nan, self.target_name, self.basedir)
    #     else:
    #         corr = self.do_area_by_area_correlation()
    #         corr_list = []
    #         for target in self.target_name:
    #             corr_list.append(corr[target])
    #         corr_direction_preference_plot(self.animal, corr_list, np.nan, self.target_name, self.basedir)


    # def rmae(self):
    #     if self.animal!='human':
    #         RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.FLN_data, self.FLN_model, self.basedir, 'FLN', self.target_name, self.source_name, self.animal)
    #     else:
    #         RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir, 'dMRI', self.target_name, self.source_name, self.animal)
    #         # RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.basedir, 'dMRI', self.target_name, self.source_name, self.animal)
    #     if self.animal=='marmoset':
    #         # 注意这里要用source_name而不是target_name
    #         RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link = strong_weak_RMAE(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir, 'dMRI', self.source_name, self.source_name, self.animal)
    #     return RMAE_strong, RMAE_medium_strong, RMAE_medium_weak, RMAE_weak, RMAE_all_link, RMAE_non_link, RMAE_both_link


    # def do_error_analysis(self):
    #     if self.animal!='human':
    #         error_analysis(self.FLN_data, self.FLN_model, self.basedir+'/error_analysis/', 'FLN', self.target_name, self.source_name, self.animal)
    #     else:
    #         error_analysis(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir+'/error_analysis/', 'dMRI', self.target_name, self.source_name, self.animal)
    #     if self.animal=='marmoset':
    #         # 注意这里要用source_name而不是target_name
    #         error_analysis(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir+'/error_analysis_dMRI/', 'dMRI', self.source_name, self.source_name, self.animal)


    # def do_neuron_level_distance_model_data(self):
    #     neuron_level_distance_model_data(self.animal, self.ss, self.tt, self.basedir)


    # def do_get_all_corr(self):
    #     if self.animal!='human':
    #         get_all_corr(self.FLN_data, self.FLN_model, self.target_name, self.basedir, 'without_intrinsic', self.animal)
    #     else:
    #         get_all_corr(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.basedir, 'without_intrinsic', self.animal)
    #         # get_all_corr(self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, self.basedir, 'without_intrinsic', self.animal)
    #     if self.animal=='marmoset':
    #         FLN_with_intrinsic_data = get_FLN_with_intrinsic(self.animal)
    #         FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
    #         FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
    #         get_all_corr(FLN_with_intrinsic_data, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_intrinsic', self.animal)

    #         FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'connection_density')
    #         get_all_corr(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_corrected_by_connection_density_intrinsic', self.animal)

    #         FLN_with_corrected_intrinsic_data = get_FLN_with_corrected_intrinsic(self.animal, 'neuron_density')
    #         get_all_corr(FLN_with_corrected_intrinsic_data, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_corrected_by_neuron_density_intrinsic', self.animal)


    #         get_all_corr(self.DTI_data, self.symmetry_CCM, self.target_name, self.basedir, 'DTI', self.animal)
    #     if self.animal=='marmoset-V4':
    #         FLNt = pd.read_excel('../../data/marmoset-V4/FLNt_data.xlsx', index_col=0)
    #         FLNt.index = FLNt.index.astype(str)
    #         FLNt.columns = FLNt.columns.astype(str)
    #         FLN_with_intrinsic_model = self.connection_count_matrix_with_self_connection.copy()
    #         FLN_with_intrinsic_model = FLN_with_intrinsic_model.div(FLN_with_intrinsic_model.sum(axis=0), axis=1)
    #         get_all_corr(FLNt, FLN_with_intrinsic_model, self.target_name, self.basedir, 'with_intrinsic', self.animal)

    
    # def do_occupied_connection(self):
    #     if self.animal!='human':
    #         occupied_connection(self.animal, self.FLN_data, self.FLN_model, self.basedir)
    #     else:
    #         occupied_connection(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.basedir)
    #         # occupied_connection(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.basedir)


    # def do_visualize_connection_matrix_and_distribution(self):
    #     if self.animal!='human':
    #         visualize_connection_matrix_and_distribution(self.animal, self.FLN_data, self.FLN_model, self.target_name, self.source_name, self.basedir, self.distance_matrix_2D, '2D')
    #         visualize_connection_matrix_and_distribution(self.animal, self.FLN_data, self.FLN_model, self.target_name, self.source_name, self.basedir, self.distance_matrix_3D, '3D')
    #     else:
    #         visualize_connection_matrix_and_distribution(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.source_name, self.basedir, self.distance_matrix_2D, '2D')
    #         visualize_connection_matrix_and_distribution(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name, self.source_name, self.basedir, self.distance_matrix_3D, '3D')
    #         # visualize_connection_matrix_and_distribution(self.animal, self.DTI_data/np.sum(self.DTI_data.values)*20, self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*20, self.target_name, self.source_name, self.basedir, self.distance_matrix_2D)
    #     if self.animal=='marmoset':
    #         # 对于marmoset，单独再画一个DTI的EDR
    #         edr_model_data(self.source_name, self.target_name, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.animal, self.basedir+'/connection_matrix_and_distribution/EDR_comparison_DTI_2D', '2D')
    #         edr_model_data(self.source_name, self.target_name, self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.animal, self.basedir+'/connection_matrix_and_distribution/EDR_comparison_DTI_3D', '3D')
    #     if self.animal=='marmoset-V4' or self.animal=='marmoset':
    #         clustered_source_name, clustered_target_name = clustered_name(self.animal)
    #         # 定义new_source_name(其实就是对于source_name的重新排序),先放置target_name再放置其余的
    #         new_source_name = clustered_target_name.copy()
    #         for area in clustered_source_name:
    #             if area not in new_source_name:
    #                 new_source_name.append(area)

    #         if self.animal=='marmoset':
    #             FLNt_data = get_FLN_with_intrinsic(self.animal)
    #         else:
    #             FLNt_data = pd.read_excel('../../data/marmoset-V4/FLNt_data.xlsx', index_col=0)
    #         FLNt_data.index = FLNt_data.index.astype(str)
    #         FLNt_data.columns = FLNt_data.columns.astype(str)

    #         FLNt_model = self.connection_count_matrix_with_self_connection.copy()
    #         FLNt_model = FLNt_model.div(FLNt_model.sum(axis=0), axis=1)

    #         vmax_dict = {'mouse': 0.2, 'macaque': 0.2, 'marmoset': 0.2, 'human': 0.2, 'marmoset-V4': 0.2}
    #         vmax = vmax_dict[self.animal]

    #         uf.connection_matrix_heatmap_vertical(FLNt_data, FLNt_model, vmax, 'jet', clustered_target_name, new_source_name, self.basedir+'/connection_matrix_and_distribution/weighted_connection_matrix_t_full', 'normal', self.animal)


    # def do_area_by_area_correlation(self):
    #     '''remove the diagonal of FLN matrix and calculate the correlation'''
    #     if self.animal=='marmoset':
    #         # 注意要对source_name做操作
    #         _ = area_by_area_correlation_without_diagonal(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.source_name)
    #     if self.animal!='human':
    #         corr = area_by_area_correlation_without_diagonal(self.FLN_data[self.target_name], self.FLN_model[self.target_name], self.target_name)
    #     else:
    #         corr = area_by_area_correlation_without_diagonal(self.DTI_data/np.sum(self.DTI_data.values)*len(self.source_name), self.symmetry_CCM/np.sum(self.symmetry_CCM.values)*len(self.source_name), self.target_name)
    #     return corr
    

    # def do_connection_proportion_wrt_to_distance_neuron_level(self):
    #     connection_proportion_wrt_to_distance_neuron_level(self.ss, self.tt, np.max(self.distance_matrix_2D.values), self.basedir)


    # def do_FLN_symmetry(self):
    #     if self.animal in ['mouse', 'marmoset', 'macaque']:
    #         FLN_symmetry(self.animal, self.FLN_data, self.FLN_model, self.basedir)


# endregion


# region process
def connection_pair_to_connection_matrix(source_idx, target_idx, area_num=None):
    '''
    将连接对转换为连接矩阵的Numpy实现
    
    参数：
    source_idx -- 源节点索引列表(一维数组,内部为整数)
    target_idx -- 目标节点索引列表(一维数组,内部为整数)
    area_num -- 可选参数，脑区数量(默认为最大索引值+1)
    
    返回：
    连接数矩阵(二维numpy数组)
    '''
    # 自动计算节点总数
    if area_num is None:
        area_num = max(np.max(source_idx), np.max(target_idx)) + 1
    
    # 初始化矩阵
    matrix = np.zeros((area_num, area_num), dtype=np.int32)
    
    # 使用np.add.at进行向量化计数
    np.add.at(matrix, (source_idx, target_idx), 1)
    
    return matrix


def area_name_to_index_array(area_name_list, ordered_area_name):
    '''
    将脑区名字列表转换为索引数组
    '''
    # 创建脑区名到索引的映射字典
    area_to_index = {area: idx for idx, area in enumerate(ordered_area_name)}
    # 将原始数组中的每个脑区名转换为对应的索引
    index_array = [area_to_index[area] for area in area_name_list]
    return index_array


def convert_split_df_to_normal(split_df, animal, dataset_key, method='sum'):
    '''
    将以分裂的脑区作为column和row的df转换为正常的
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    full_name = animal_info.full_name
    full_split_name = animal_info.full_split_name
    split_name_to_name_dict = animal_info.split_name_to_name_dict

    df = cf.get_zero_df(index=full_name, columns=full_name)
    count_split_df = cf.get_zero_df(index=full_name, columns=full_name)
    for row in full_split_name:
        for col in full_split_name:
            merge_row = split_name_to_name_dict[row]
            merge_col = split_name_to_name_dict[col]
            df.loc[merge_row, merge_col] += split_df.loc[row, col]
            count_split_df.loc[merge_row, merge_col] += 1
    if method == 'sum':
        pass
    elif method == 'mean':
        df = df / count_split_df
    return df


def broadcast_value_to_split_area(value_dict, animal, dataset_key):
    '''
    将value_dict中的值广播到每个split_area(比如某个区域有a--1,a--2,则将value_dict['a']广播到a--1和a--2,获得和原先一样的数值)
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    split_value_dict = {}
    for area in animal_info.full_name:
        if area in value_dict:
            for split_area in animal_info.name_to_split_name_dict[area]:
                split_value_dict[split_area] = value_dict[area]
    return split_value_dict


def assign_value_to_split_area(split_value_dict, key, value, animal, dataset_key):
    '''
    对split_value_dict中的某个key重新赋值,这个key或许不存在于split_value_dict中,因为在里面的可能是分开的多个
    这种赋值模式保证覆盖到所有split_area
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    for split_area in animal_info.name_to_split_name_dict[key]:
        split_value_dict[split_area] = value
    return split_value_dict


def CCM_to_FLN(CCM, animal, dataset_key):
    '''
    将CCM转换为FLN(FLN是指以target正则化所有输入连接的矩阵)
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    FLN = cf.normalize_df_with_sum(CCM, row_or_col='col')
    return FLN


def normalize_SC(SC, animal, dataset_key):
    '''
    如果SC是Dmri数据,需要将其正则化
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    if animal_info.SC_type == 'Dmri':
        raise NotImplementedError
    return SC


def scale_SC_to_int(SC, minimal_tol=None):
    '''
    将SC数据缩放到整数

    由于SC可能有极小的值,按照此值来scale可能会得到过大的int,但是这些值对于后续分布的影响极小,在快速计算的情况下,可以设定minimal_tol
    '''
    minimal_value = np.min(SC.values)
    if minimal_tol is not None:
        if minimal_value < minimal_tol:
            minimal_value = minimal_tol
    SC_int = (SC / minimal_value).astype(int)
    return SC_int


def weight_matrix_to_topo_matrix(weight_matrix):
    '''
    将权重矩阵转化为拓扑矩阵
    '''
    if isinstance(weight_matrix, pd.DataFrame):
        # 如果输入是DataFrame，将其转换为NumPy数组进行处理
        topo_matrix = (weight_matrix.values > 0).astype(int)
        return pd.DataFrame(topo_matrix, index=weight_matrix.index, columns=weight_matrix.columns)
    
    elif isinstance(weight_matrix, np.ndarray):
        # 如果输入是NumPy数组，直接处理
        return (weight_matrix > 0).astype(int)
    
    else:
        raise ValueError("输入必须是 pandas.DataFrame 或 numpy.ndarray 类型")


def prioritize_sublist(l, sub_l):
    '''
    将sub_l中的元素放在l的前面,其余按顺序排列
    '''
    back = [item for item in l if item not in sub_l]
    return sub_l + back


def rearrange_SC(SC, animal, dataset_key):
    '''
    重排列SC
    rearrange: 是否重新排列SC的顺序,使得被注射的target区域排在前面,其余按顺序排列
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    reordered_full_name = animal_info.reordered_full_name
    new_SC = cf.reorder_df(SC, reordered_full_name, reordered_full_name)
    return new_SC


def log_SC(SC, animal, dataset_key, base=10, apply_c=False, apply_w=False):
    '''
    对SC取log
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    c = animal_info.c
    w = animal_info.w

    SC_log = np.log(SC) / np.log(base)
    # 处理nan,inf,-inf
    SC_log = SC_log.replace([np.inf, -np.inf], np.nan)
    if apply_c:
        SC_log = SC_log + c
    if apply_w:
        # 将-inf替换为w
        SC_log = SC_log.replace(-np.inf, w)
        # 将小于w的值设为w
        SC_log[SC_log < w] = w
    return SC_log



# endregion


# region analyze
def angle_distance_independence_test():
    pass


def SC_symmetry_index():
    pass


def connection_proportion_wrt_to_distance_neuron_level():
    pass


def residual_heterogeneity():
    '''
    可以先用某种方式regress掉skew ness对于performance的影响,然后看看剩下的?
    '''
    pass
# endregion


# region visualization
def show_value_on_flatmap(ax, animal, dataset_key, value_dict, cmap, cbar_label='', target=None, source=None, show_area_name=False, text_fontsize=cf.FONT_SIZE/3, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, edgecolor=cf.BLACK, cbar=True, cbar_position=None, fill=True, adjust_lim=True, polygon_kwargs=None, cbar_kwargs=None):
    '''
    在flatmap上显示value_dict的值,有两种使用方式,指定某个target或者source以specific_color显示,其余值用cmap显示,如果不指定target和source,则所有值用cmap显示
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    name_location_dict = animal_info.name_location_dict
    coords_dict = animal_info.coords_dict

    split_value_dict = broadcast_value_to_split_area(value_dict, animal, dataset_key)
    mask = {area: False for area in split_value_dict.keys()}
    mask_color = cf.TRANSPARENT
    if target is not None:
        mask = assign_value_to_split_area(mask, key=target, value=True, animal=animal, dataset_key=dataset_key)
        mask_color = TARGET_COLOR
    if source is not None:
        mask = assign_value_to_split_area(mask, key=source, value=True, animal=animal, dataset_key=dataset_key)
        mask_color = SOURCE_COLOR
    cf.plt_polygon_heatmap(ax, xy_dict=coords_dict, value_dict=split_value_dict, mask=mask, mask_color=mask_color, cbar_label=cbar_label, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, edgecolor=edgecolor, cbar=cbar, cbar_position=cbar_position, fill=fill, adjust_lim=adjust_lim, polygon_kwargs=polygon_kwargs, cbar_kwargs=cbar_kwargs)

    if show_area_name:
        for name, location in name_location_dict.items():
            cf.add_text(ax, x=location[0], y=location[1], text=name, fontsize=text_fontsize, ha='center', va='center', transform='data')

    ax.axis('off')
    ax.axis('equal')


def show_area_boundary(ax, animal, dataset_key, show_area_name=False, text_fontsize=cf.FONT_SIZE/3, edgecolor=cf.BLACK):
    '''
    在flatmap上画出脑区边界
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    value_dict = {area: 0 for area in animal_info.full_name}
    cmap = cf.get_cmap([cf.TRANSPARENT])
    show_value_on_flatmap(ax, animal, dataset_key, value_dict=value_dict, show_area_name=show_area_name, text_fontsize=text_fontsize, edgecolor=edgecolor, fill=False, cbar=False, cmap=cmap)

@cf.message_decorator('需要兼容可能的顺行失踪')
def draw_rectangle_for_connection_matrix(ax, animal, dataset_key):
    '''
    在连接矩阵上画出矩形,用于区分实验已知部分和位置部分
    '''
    color = cf.GREEN
    linestyle = '--'
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    target_num = len(target_name)
    source_num = len(source_name)
    cf.plt_line(ax, x=[0, target_num], y=[0, 0], color=color, linestyle=linestyle)
    cf.plt_line(ax, x=[0, 0], y=[0, source_num], color=color, linestyle=linestyle)
    cf.plt_line(ax, x=[target_num, target_num], y=[0, source_num], color=color, linestyle=linestyle)
    cf.plt_line(ax, x=[0, target_num], y=[source_num, source_num], color=color, linestyle=linestyle)

@cf.message_decorator('需要兼容可能的顺行失踪')
def draw_mask_for_unmeasured_block(ax, target_name, source_name):
    '''
    在连接矩阵上画出mask,用于区分未测量部分
    '''
    target_num = len(target_name)
    source_num = len(source_name)
    if target_num < source_num:
        color = cf.GRAY
        cmap = cf.get_cmap([color])
        unmeasured_matrix = np.zeros((source_num, source_num))
        for i in range(source_num):
            if i < target_num:
                unmeasured_matrix[:, i] = np.nan
            else:
                unmeasured_matrix[:, i] = 1
        cf.sns_heatmap(ax, unmeasured_matrix, show_xtick=False, show_ytick=False, cmap=cmap, cbar=False)
        cf.add_text(ax, x=(target_num+source_num)/2, y=source_num/2, text='Unknown', fontsize=cf.LABEL_SIZE, ha='center', va='center', transform='data')


def visualize_connection_matrix_heatmap(ax_data, ax_model, matrix_data, matrix_model, animal, dataset_key, log_matrix=False, topo=False):
    '''
    log_matrix: 是否对矩阵取log
    topo: 是否将矩阵转化为拓扑矩阵
    '''
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    SC_type = animal_info.SC_type

    if log_matrix:
        c = get_log_c(animal, SC_type)
        w = get_log_w(animal, SC_type)
        local_matrix_data = log_SC(SC=matrix_data, animal=animal, SC_type=SC_type, apply_c=False, apply_w=False)
        local_matrix_model = log_SC(SC=matrix_model, animal=animal, SC_type=SC_type, apply_c=False, apply_w=False)
        vmin = w
        vmax = c
        add_geq = False

    else:
        local_matrix_data = matrix_data
        local_matrix_model = matrix_model
        vmin = 0.
        vmax = get_connection_matrix_vmax(animal, SC_type)
        add_geq = True

    if topo:
        cmap = plt.cm.hot
        local_matrix_data = weight_matrix_to_topo_matrix(local_matrix_data)
        local_matrix_model = weight_matrix_to_topo_matrix(local_matrix_model)
        vmin = 0.
        vmax = 1.
    else:
        cmap = plt.cm.jet
    ticks = [vmin, vmax]

    # 连接矩阵
    cbar_kwargs = {'add_geq': add_geq, 'ticks': ticks}
    cf.sns_heatmap(ax_data, local_matrix_data, show_xtick=False, show_ytick=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)
    cf.sns_heatmap(ax_model, local_matrix_model, show_xtick=False, show_ytick=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs)

    # 未测量部分
    draw_mask_for_unmeasured_block(ax_data, target_name, source_name)
    
    # 用框指示已测量部分
    draw_rectangle_for_connection_matrix(ax_data, target_name, source_name)
    draw_rectangle_for_connection_matrix(ax_model, target_name, source_name)

    cf.set_ax(ax_data, title=animal+'\nData', xlabel='Target', ylabel='Source')
    cf.set_ax(ax_model, title='Model', xlabel='Target', ylabel='Source')

    matrix_data_to_compare = cf.get_sub_df(matrix_data, col_list=target_name)
    matrix_model_to_compare = cf.get_sub_df(matrix_model, col_list=target_name)
    corr = cf.get_pd_rectangle_corr(matrix_data_to_compare, matrix_model_to_compare, diagonal=False)
    cf.add_text(ax_model, x=0.5, y=1.25, text='Corr='+str(round(corr, 3)), va='center', ha='center', fontsize=cf.LABEL_SIZE)

@cf.message_decorator('需要兼容可能的顺行失踪')
def visualizae_multiple_connection_matrix_heatmap(animal_list, model_dir_list, SC_type_list, figname):
    '''
    同时画多个动物的连接矩阵
    '''
    animal_num = len(animal_list)
    fig, ax = cf.gfa(ncols=animal_num, nrows=2)
    for i, animal, model_dir, SC_type in zip(range(animal_num), animal_list, model_dir_list, SC_type_list):
        SC_data = load_SC_data(animal, SC_type=SC_type, rearrange=True, cluster_name=True)
        SC_model = load_SC_model(animal, SC_type=SC_type, basedir=model_dir, rearrange=True, cluster_name=True)
        visualize_connection_matrix_heatmap(ax_data=ax[0][i], ax_model=ax[1][i], matrix_data=SC_data, matrix_model=SC_model, animal=animal, SC_type=SC_type, log_matrix=False, topo=False)
    cf.add_axes_list_tag_by_order(ax[0, :])
    cf.save_fig(fig, figname, formats=['pdf', 'png'], bbox_inches='tight')


def visualize_distribution_scaled_by_SC(ax, some_property, SC, minimal_tol=None, distribution_type='CDF', distribution_kwargs=None):
    '''
    利用SC来scale得到property的比例
    '''
    SC_int = scale_SC_to_int(SC, minimal_tol=minimal_tol)
    repeated_property = cf.repeat_data(some_property, SC_int)
    distribution_kwargs = cf.update_dict({}, distribution_kwargs)
    if distribution_type == 'CDF':
        cf.plt_cdf(ax, repeated_property, SC_int, **distribution_kwargs)
    elif distribution_type == 'hist':
        cf.plt_hist(ax, repeated_property, SC_int, **distribution_kwargs)


class SingleVisualizer:
    '''
    进行单个动物的可视化
    '''
    def __init__(self, animal, dataset_key, model_dir):
        self.animal = animal
        self.dataset_key = dataset_key
        self.animal_info = AnimalInfo(animal, dataset_key)
        self.SC = self.animal_info.SC
        self.model_dir = model_dir
        self.minimal_tol = 1e-3 # 后续改成高精度

    def connection_on_flatmap(self, ax):
        pass

    def vector_field(self, ax):
        pass

    def model_data_SC(self, ax_model, ax_data):
        pass

    def area_level_performance(self, ax, performance_metric='correlation'):
        pass

    def distance_distribution(self, ax, dim='2D'):
        if dim == '2D':
            distance_matrix = self.animal_info.distance_2D
        elif dim == '3D':
            distance_matrix = self.animal_info.distance_3D
        
        visualize_distribution_scaled_by_SC(ax, distance_matrix, self.SC, minimal_tol=self.minimal_tol, distribution_type='hist', distribution_kwargs=None)

    def angle_distribution(self, ax, dim='2D'):
        if dim == '2D':
            angle_matrix = self.animal_info.angle_2D
            visualize_distribution_scaled_by_SC(ax, angle_matrix, self.SC, minimal_tol=self.minimal_tol, distribution_type='CDF', distribution_kwargs=None)
        elif dim == '3D':
            angle_matrix = self.animal_info.angle_3D

    def SC_distance_scatter(self, ax, dim='2D'):
        pass

    def SC_distribution(self, ax, log_transform=True):
        pass

    def retrograde_neuron(self, ax):
        pass

    def rm_long_range_correlation_change(self, ax, rm_mode='data'):
        pass

    def category_long_range_proportion(self, ax):
        pass


class MultipleVisualizer:
    '''
    进行多个动物的可视化
    '''
    def __init__(self, animal_list, dataset_key_list, model_dir_list):
        pass


class PaperFig:
    def __init__(self):
        pass

    def figure_1(self):
        '''
        生成模型示意图
        '''
        animal = 'marmoset'
        dataset_key = 'majka'
        animal_info = AnimalInfo(animal, dataset_key)
        polygon_dict = animal_info.polygon_dict
        full_name = animal_info.full_name
        area_center_dict = animal_info.area_center_dict
        whole_cortex_polygon = animal_info.whole_cortex_polygon
        point_density = 15
        source_area = 'V3'
        target_area = 'V1'
        source_center = area_center_dict[source_area]
        target_center = area_center_dict[target_area]
        Lambda = 2
        exp_x = np.linspace(0, 10, 1000)
        exp_y = np.exp(-exp_x/Lambda)/Lambda
        uniform_x = np.linspace(-np.pi, np.pi, 1000)
        uniform_y = np.ones(uniform_x.shape) / (2*np.pi)
        margin = cf.COMPACT_MARGIN
        margin['top'] = 0.87
        margin['bottom'] = 0.13

        fig, axes = cf.get_fig_ax(ncols=7, nrows=2, margin=cf.COMPACT_MARGIN, ax_width=cf.AX_WIDTH*0.6, ax_height=cf.AX_HEIGHT*0.6)

        cf.add_axes_list_tag_by_order([axes[0, 0], axes[0, 2], axes[0, 4], axes[0, 6], axes[1, 6]])

        ax_dict = {}
        ax_dict['a'] = cf.merge_ax(axes[0:2, 0:2])
        ax_dict['b'] = cf.merge_ax(axes[0:2, 2:4])
        ax_dict['c'] = cf.merge_ax(axes[0:2, 4:6])
        ax_dict['d'] = axes[0, 6]
        ax_dict['e'] = axes[1, 6]

        for ax in [ax_dict['a'], ax_dict['b'], ax_dict['c']]:
            show_area_boundary(ax, animal, dataset_key)

        # colors = cf.get_random_color(n=len(full_name), cmap=cf.HEATMAP_CMAP, seed=3)
        colors = cf.get_plt_color(n=len(full_name))
        color_dict = cf.create_dict(full_name, colors)
        split_color_dict = broadcast_value_to_split_area(color_dict, animal, dataset_key)

        ax = ax_dict['b']
        s = cf.get_suitable_s(ax, num=int(point_density*whole_cortex_polygon.area)) / 10
        for area, polygon in polygon_dict.items():
            points = generate_random_point_in_polygon(polygon, density=point_density, seed=0)
            cf.plt_scatter(ax, points[:, 0], points[:, 1], color=split_color_dict[area], s=s, zorder=-1)

        ax = ax_dict['c']
        for area, polygon in polygon_dict.items():
            if area in [source_area, target_area]:
                points = generate_random_point_in_polygon(polygon, density=point_density, seed=0)
                cf.plt_scatter(ax, points[:, 0], points[:, 1], color=split_color_dict[area], s=s, zorder=-1)
        cf.add_arrow(ax, x_start=source_center[0], y_start=source_center[1], x_end=target_center[0], y_end=target_center[1])

        ax = ax_dict['d']
        cf.plt_line(ax, exp_x, exp_y)
        cf.add_text(ax, text='$EXP(\Lambda)\sim \\frac{1}{\Lambda} e^{-\\frac{d}{\Lambda}}$', x=0.5, y=0.8, va='center', ha='center')
        cf.set_ax(ax, xlabel='d', ylabel='p(d)', title='EDR', text_process={'capitalize': False})
        cf.rm_ax_tick(ax)
        cf.rm_ax_ticklabel(ax)

        ax = ax_dict['e']
        cf.plt_line(ax, uniform_x, uniform_y)
        cf.add_text(ax, text='$U(-\pi, \pi)\sim \\frac{1}{2\pi}$', x=0.5, y=0.8, va='center', ha='center')
        cf.set_ax(ax, xlabel='$\\theta$', ylabel='p($\\theta$)', title='UAR')
        cf.rm_ax_tick(ax)
        cf.rm_ax_ticklabel(ax)

    def figure_2(self):
        pass

    def figure_3(self):
        pass

    def figure_4(self):
        pass

    def figure_5(self):
        pass

    def figure_6(self):
        pass

    def figure_7(self):
        pass
# endregion