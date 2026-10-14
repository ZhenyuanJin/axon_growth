import abc
import copy
import datetime
import glob
import hashlib
import inspect
import multiprocessing
import os
import pickle
import re
import shutil
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps

import joblib
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.stats as st
import seaborn as sns
import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.smoothers_lowess import lowess

from .. import adjust_text_custom

CF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLUE = (0.01, 0.45, 0.7)
RED = (1.0, 0.27, 0.01)
GREEN = (0.01, 0.65, 0.5)
YELLOW = (0.99, 0.7, 0.01)
BLACK = (0.01, 0.01, 0.01)
WHITE = (1.0, 1.0, 1.0)
GRAY = (0.5, 0.5, 0.5)
ORANGE = (1.0, 0.5, 0.01)
PURPLE = (0.5, 0.01, 0.5)
CRIMSON = (220 / 255, 20 / 255, 60 / 255)
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
RANA = (235 / 255, 235 / 255, 235 / 255)
RYO = (93 / 255, 116 / 255, 165 / 255)
MARGIN = {'left': 0.2, 'right': 0.85, 'bottom': 0.2, 'top': 0.85}
COMPACT_MARGIN = {'left': 0.1, 'right': 0.95, 'bottom': 0.1, 'top': 0.95}
ADJUST_PARAMS_CUSTOM = {'left': 0.2, 'right': 0.85, 'bottom': 0.2, 'top': 0.85, 'wspace': 0.35, 'hspace': 0.35}
FONT_SIZE = 15
TITLE_SIZE = FONT_SIZE * 2
SUP_TITLE_SIZE = FONT_SIZE * 3
LABEL_SIZE = FONT_SIZE * 2
LABEL_PAD = LABEL_SIZE / 3
TITLE_PAD = TITLE_SIZE / 3
TICK_SIZE = FONT_SIZE * 4 / 3
LEGEND_SIZE = FONT_SIZE * 4 / 3
LINE_WIDTH = 3
MARKER_SIZE = LINE_WIDTH * 2
LEGEND_FRAME_LINEWIDTH = LINE_WIDTH / 3
ERRORBAR_LINEWIDTH = LINE_WIDTH
AX_WIDTH = 5
AX_HEIGHT = 5
FIG_SIZE = (AX_WIDTH / (MARGIN['right'] - MARGIN['left']), AX_HEIGHT / (MARGIN['top'] - MARGIN['bottom']))
FIG_DPI = 300
SAVEFIG_DPI = 300
SAVEFIG_VECTOR_FORMAT = 'pdf'
SAVEFIG_RASTER_FORMAT = 'png'
SAVEFIG_FORMAT = SAVEFIG_VECTOR_FORMAT
TOP_SPINE = False
RIGHT_SPINE = False
LEFT_SPINE = True
BOTTOM_SPINE = True
LEGEND_LOC = 'upper right'
MATHTEXT_FONTSET = 'stix'
PDF_FONTTYPE = 42
PS_FONTTYPE = 42
PDF_USE14COREFONTS = False
PAD_INCHES = 0.2
USE_MATHTEXT = True
USE_OFFSET = False
AXES_LINEWIDTH = LINE_WIDTH
TICK_MAJOR_WIDTH = 2
TICK_MINOR_WIDTH = 1
TICK_MAJOR_SIZE = 8
TICK_MINOR_SIZE = 4
TICK_DIRECTION = 'out'
RM_REPEAT_TICK_LABEL_WHEN_SHARE = True
AX_FACECOLOR = (1.0, 1.0, 1.0, 0.9)
plt.rcParams['figure.figsize'] = FIG_SIZE
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_SIZE
plt.rcParams['axes.titlesize'] = TITLE_SIZE
plt.rcParams['axes.labelpad'] = LABEL_PAD
plt.rcParams['axes.titlepad'] = TITLE_PAD
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['legend.fontsize'] = LEGEND_SIZE
plt.rcParams['axes.spines.top'] = TOP_SPINE
plt.rcParams['axes.spines.right'] = RIGHT_SPINE
plt.rcParams['axes.spines.left'] = LEFT_SPINE
plt.rcParams['axes.spines.bottom'] = BOTTOM_SPINE
plt.rcParams['axes.linewidth'] = AXES_LINEWIDTH
plt.rcParams['lines.linewidth'] = LINE_WIDTH
plt.rcParams['lines.markersize'] = MARKER_SIZE
plt.rcParams['patch.linewidth'] = LEGEND_FRAME_LINEWIDTH
plt.rcParams['savefig.format'] = SAVEFIG_FORMAT
plt.rcParams['figure.dpi'] = FIG_DPI
plt.rcParams['savefig.dpi'] = SAVEFIG_DPI
plt.rcParams['legend.loc'] = LEGEND_LOC
plt.rcParams['mathtext.fontset'] = MATHTEXT_FONTSET
plt.rcParams['pdf.fonttype'] = PDF_FONTTYPE
plt.rcParams['ps.fonttype'] = PS_FONTTYPE
plt.rcParams['pdf.use14corefonts'] = PDF_USE14COREFONTS
plt.rcParams['savefig.pad_inches'] = PAD_INCHES
plt.rcParams['axes.formatter.use_mathtext'] = USE_MATHTEXT
plt.rcParams['axes.formatter.useoffset'] = USE_OFFSET
plt.rcParams['axes.linewidth'] = LINE_WIDTH
plt.rcParams['xtick.major.width'] = TICK_MAJOR_WIDTH
plt.rcParams['xtick.minor.width'] = TICK_MINOR_WIDTH
plt.rcParams['ytick.major.width'] = TICK_MAJOR_WIDTH
plt.rcParams['ytick.minor.width'] = TICK_MINOR_WIDTH
plt.rcParams['xtick.major.size'] = TICK_MAJOR_SIZE
plt.rcParams['xtick.minor.size'] = TICK_MINOR_SIZE
plt.rcParams['ytick.major.size'] = TICK_MAJOR_SIZE
plt.rcParams['ytick.minor.size'] = TICK_MINOR_SIZE
plt.rcParams['xtick.direction'] = TICK_DIRECTION
plt.rcParams['ytick.direction'] = TICK_DIRECTION
plt.rcParams['axes.facecolor'] = AX_FACECOLOR
plt.rcParams['figure.subplot.wspace'] = ADJUST_PARAMS_CUSTOM['wspace']
plt.rcParams['figure.subplot.hspace'] = ADJUST_PARAMS_CUSTOM['hspace']
fm.fontManager.addfont(os.path.join(CF_DIR, 'font/arial.ttf'))
plt.rcParams['font.family'] = 'Arial'
SEED = 421
ROUND_DIGITS = 3
ROUND_FORMAT = 'general'
INF_POLICY = 'to_nan'
NAN_POLICY = 'drop'
TEXT_PROCESS = {'capitalize': True, 'replace_underscore': ' ', 'ignore_equation_underscore': True}
REPLACE_DOT = 'd'
FILENAME_PROCESS = {'replace_blank': '_', 'replace_dot': REPLACE_DOT}
BAR_WIDTH = 0.8
BIN_NUM = 20
XTICK_ROTATION = 90
YTICK_ROTATION = 0
AUXILIARY_LINE_STYLE = '--'
PLT_CAP_SIZE_RATIO = 1.0
PLT_CAP_SIZE = LINE_WIDTH * PLT_CAP_SIZE_RATIO
ARROW_STYLE = '-|>'
ARROW_HEAD_WIDTH = 0.3
ARROW_HEAD_LENGTH = 0.5
ARROW_PROPS = {'fc': RED, 'ec': RED, 'linewidth': LINE_WIDTH, 'arrowstyle': ARROW_STYLE, 'head_length': ARROW_HEAD_LENGTH, 'head_width': ARROW_HEAD_WIDTH}
FAINT_ALPHA = 0.5
TEXT_X = 0.05
TEXT_Y = 0.95
ELEV = 30
AZIM = 30
CBAR_LABEL_SIZE = TICK_SIZE
CBAR_TICK_SIZE = TICK_SIZE
MASK_COLOR = GRAY
TEXT_VA = 'top'
TEXT_HA = 'left'
BBOX_INCHES = None
SIDE_PAD = 0.03
CBAR_POSITION = {'position': 'right', 'size': 0.05, 'pad': SIDE_PAD}
CBAR_POSITION_3D = {'position': 'right', 'size': 0.05, 'pad': SIDE_PAD * 6}
TICK_PROPORTION = 0.9
SAVEFIG_PKL = False
CMAP = plt.cm.viridis
HEATMAP_CMAP = plt.cm.jet
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list('density_cmap', [RANA, BLUE])
TAG_CASE = 'lower'
TAG_PARENTHESES = False
AX_TAG_POS = (-0.2, 1.2)
TAG_SIZE = SUP_TITLE_SIZE
TAG_VA = TEXT_VA
TAG_HA = TEXT_HA
PRINT_WIDTH = 80
PRINT_CHAR = '-'

def get_local_rng(seed):
    return np.random.RandomState(seed)

def get_time_seed():
    timestamp = int(time.time())
    microsecond = datetime.datetime.now().microsecond
    return (timestamp * 1000000 + microsecond) % (2 ** 31 - 1)

def get_start_time():
    return time.perf_counter()

def _readable_time(seconds):
    if seconds < 60:
        return f'{seconds:.9f} s'
    elif seconds < 3600:
        return f'{seconds / 60:.2f} min'
    elif seconds < 86400:
        return f'{seconds / 3600:.2f} h'
    else:
        return f'{seconds / 86400:.2f} d'

def get_interval_time(start, title='', print_info=True, **kwargs):
    interval = time.perf_counter() - start
    if print_info:
        if 'digits' in kwargs:
            print('do not use digits, this will be deprecated soon')
        if 'format_type' in kwargs:
            print('do not use format_type, this will be deprecated soon')
        print(f'{title} takes {_readable_time(interval)}')
    return interval

def func_timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = get_start_time()
        result = func(*args, **kwargs)
        interval = get_interval_time(start, title=func.__name__, print_info=True)
        return result
    return wrapper

def flex_print_title(print_func=print, title=None, n=PRINT_WIDTH, char=PRINT_CHAR):
    if title is not None:
        title = f'{title}'
        length_start = (n - len(title)) // 2
        length_end = n - len(title) - length_start
        print_func(length_start * char + title + length_end * char)
    else:
        print_func(char * n)

def print_title(title, n=PRINT_WIDTH, char=PRINT_CHAR):
    flex_print_title(print_func=print, title=title, n=n, char=char)

def method_decorator(decorator):

    def decorate(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, staticmethod):
                setattr(cls, attr_name, staticmethod(decorator(attr_value.__func__)))
            elif isinstance(attr_value, classmethod):
                setattr(cls, attr_name, classmethod(decorator(attr_value.__func__)))
            elif callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls
    return decorate

def message_decorator(message, print_func=print):

    def decorator(obj):
        if inspect.isfunction(obj):

            @wraps(obj)
            def wrapper(*args, **kwargs):
                print_func(message.format(obj.__name__))
                return obj(*args, **kwargs)
            return wrapper
        elif inspect.isclass(obj):
            for name, method in obj.__dict__.items():
                if callable(method):
                    if name == '__init__':
                        message_formatted_by_cls_name = message.format(obj.__name__)
                        setattr(obj, name, message_decorator(message=message_formatted_by_cls_name, print_func=print_func)(method))
            return obj
        else:
            raise TypeError('This decorator supports only classes and functions.')
    return decorator

def to_be_improved(obj):
    return message_decorator("The '{0}' needs improvement.")(obj)

def func_tracker(func, print_func=print_title):
    return message_decorator('Running: {0}', print_func=print_func)(func)

def init_tracker(func, print_func=print_title, cls_name=None):
    return message_decorator(f'Instance of {cls_name} created', print_func=print_func)(func)

def cls_tracker(cls):
    for name, method in cls.__dict__.items():
        if callable(method):
            if name == '__init__':
                setattr(cls, name, init_tracker(method, cls_name=cls.__name__))
            else:
                setattr(cls, name, func_tracker(method))
    return cls

def tracker_with_time(obj):
    if inspect.isfunction(obj):
        return func_timer(func_tracker(obj))
    elif inspect.isclass(obj):
        return method_decorator(func_timer)(cls_tracker(obj))

def get_detailed_error():
    exc_type, exc_value, exc_trace = sys.exc_info()
    if not exc_type:
        return
    error_name = exc_type.__name__
    error_msg = str(exc_value)
    stack_summary = traceback.extract_tb(exc_trace)
    last_frame = stack_summary[-1] if stack_summary else None
    error_info = f"======== ======== :{error_name}:{error_msg}:{(last_frame.filename if last_frame else 'unknown')}:{(last_frame.lineno if last_frame else 0)}:{(last_frame.line if last_frame else 'unknown')}======== ========{traceback.format_exc()}\n    "
    return error_info

class FlexibleTry:

    def __init__(self, enable_try=True, print_info=True):
        self.enable_try = enable_try
        self.print_info = print_info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
            self.exception = None
            return False
        self.success = False
        self.exception = exc_val
        if self.enable_try and exc_type:
            error_info = get_detailed_error()
            if self.print_info:
                print(error_info)
            return True
        return False

def caller_filename():
    stack = inspect.stack()
    caller_frame = stack[1]
    caller_filename = caller_frame.filename
    return caller_filename

def current_file():
    stack = inspect.stack()
    caller_frame = stack[1]
    return caller_frame.filename

def safe_path_join(*paths):
    cleaned_paths = []
    for i, path in enumerate(paths):
        if path is None or path == '':
            continue
        elif i == 0:
            cleaned_paths.append(path.rstrip('/\\'))
        else:
            cleaned_paths.append(path.strip('/\\'))
    return os.path.join(*cleaned_paths)

def pj(*args):
    return safe_path_join(*args)

def mkdir(fn):
    os.makedirs(fn, exist_ok=True)

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"'{file_path}' .")
    except FileNotFoundError:
        print(f": '{file_path}' .")
    except PermissionError:
        print(f": '{file_path}'.")
    except Exception as e:
        print(f':{e}')

def find_files_with_any_extension(path):
    result = []
    if os.path.exists(path):
        result.append(path)
    dir_path = os.path.dirname(path) or '.'
    file_name = os.path.basename(path)
    pattern = os.path.join(dir_path, f'{file_name}.*')
    for p in glob.glob(pattern):
        if os.path.isfile(p) and (not p.endswith('.')):
            if p != path:
                result.append(p)
    return result

def check_all_file_exist_with_any_extension(*paths):
    for path in paths:
        if not find_files_with_any_extension(path):
            return False
    return True

def get_subfile(basedir, full=True):
    if full:
        l = [os.path.join(basedir, f) for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]
    else:
        l = [f for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]
    l = natsort.natsorted(l)
    return l

def save_dict_txt(filename, dict_data, key_to_save=None):
    data_to_save = dict_data if key_to_save is None else {k: dict_data[k] for k in key_to_save}
    if not filename.endswith('.txt'):
        filename += '.txt'
    mkdir(os.path.dirname(filename))
    with open(filename, 'w') as txt_file:

        def write_dict(d, indent):
            for key, value in d.items():
                if isinstance(value, dict):
                    txt_file.write(' ' * indent + f'{key}:\n')
                    write_dict(value, indent + 4)
                else:
                    txt_file.write(' ' * indent + f'{key}: {value}\n')
        write_dict(data_to_save, 0)

def save_dict(dict_data, filename, format_list=None, key_to_save=None):
    if format_list is None:
        format_list = ['txt', 'pkl']
    if key_to_save is None:
        key_to_save = list(dict_data.keys())
    mkdir(os.path.dirname(filename))
    if filename.endswith('.txt'):
        format_list.append(filename.split('.')[-1])
        filename = os.path.splitext(filename)[0]
    if filename.endswith('.pkl') or filename.endswith('.pickle') or filename.endswith('.joblib'):
        format_list.append('.pkl')
        filename = os.path.splitext(filename)[0]
    if 'txt' in format_list:
        save_dict_txt(filename, dict_data, key_to_save=key_to_save)
    if 'pkl' in format_list:
        save_pkl({k: dict_data[k] for k in key_to_save}, filename)

def get_load_function(format):
    d = {'npy': load_array, 'npz': load_sps_array, 'pkl': load_pkl, 'joblib': load_pkl, 'pickle': load_pkl, '': load_dict_separate}
    return d[format]

def get_save_function_for_object(obj):
    if isinstance(obj, np.ndarray):
        return save_array
    elif isinstance(obj, sps.spmatrix):
        return save_sps_array
    else:
        return save_pkl

def save_key_value_pair(key, value, save_func_dict, save_kwargs_dict, save_dir):
    str_key = hash_or_str(key)
    save_func = save_func_dict[key]
    save_kwargs = save_kwargs_dict[key]
    save_func(value, os.path.join(save_dir, str_key), **save_kwargs)
    return (str_key, key)

def save_dict_separate(dict_data, save_dir, save_func_dict=None, save_kwargs_dict=None, overwrite=False, save_as_txt=True, process_num=1, key_to_save=None, max_depth=0, current_depth=0, remove_after_save=False):
    if len(dict_data) == 0:
        print_title('Warning: The dictionary is empty. No files will be saved.', char='*')
        return
    if key_to_save is None:
        key_to_save = list(dict_data.keys())
    save_func_dict = fill_missing_key(save_func_dict, key_to_save, None)
    save_kwargs_dict = fill_missing_key(save_kwargs_dict, key_to_save, {})
    for k in key_to_save:
        if save_func_dict.get(k, None) is not None:
            continue
        value = dict_data[k]
        if isinstance(value, dict) and current_depth < max_depth:
            save_func_dict[k] = partial(save_dict_separate, save_as_txt=save_as_txt, max_depth=max_depth, current_depth=current_depth + 1)
        else:
            save_func_dict[k] = get_save_function_for_object(value)
    if overwrite and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    metadata = {}
    dict_data_to_save = {k: dict_data[k] for k in key_to_save}
    results = multi_process_items_for(process_num=process_num, func=save_key_value_pair, for_dict=dict_data_to_save, kwargs={'save_func_dict': save_func_dict, 'save_kwargs_dict': save_kwargs_dict, 'save_dir': save_dir}, func_name=f'save dict to {save_dir}', use_tqdm=False)
    for r in results:
        metadata[r[0]] = r[1]
    save_dict(metadata, os.path.join(save_dir, 'metadata'))
    if save_as_txt:
        for k, v in metadata.items():
            if k in key_to_save:
                save_txt(dict_data[k], os.path.join(save_dir, 'preview', v))
    if remove_after_save:
        for k in key_to_save:
            del dict_data[k]

def part_load_dict_separate(load_dir, subfile, metadata, key_to_load):
    filename, ext = os.path.splitext(subfile)
    if filename == 'metadata':
        return (None, None)
    if filename == 'preview':
        return (None, None)
    if filename in metadata.keys():
        if metadata[filename] in key_to_load:
            load_func = get_load_function(ext[1:])
            return (metadata[filename], load_func(os.path.join(load_dir, subfile)))
    return (None, None)

def _compatible_data_container(obj):

    @wraps(obj)
    def wrapper(*args, **kwargs):
        r = obj(*args, **kwargs)
        if isinstance(r, DataContainer):
            pass
        elif isinstance(r, dict):
            if '_config' in r:
                r = _dict_to_data_container(r)
        return r
    return wrapper

@_compatible_data_container
def load_dict_separate(load_dir, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    metadata = load_pkl(os.path.join(load_dir, 'metadata'))
    if key_to_load is None:
        key_to_load = list(metadata.values())
    if filter_str is not None:
        if isinstance(filter_str, str):
            filter_str = [filter_str]
        if filter_mode == 'include':
            if filter_logic == 'or':
                key_to_load = [k for k in key_to_load if any((s in k for s in filter_str))]
            elif filter_logic == 'and':
                key_to_load = [k for k in key_to_load if all((s in k for s in filter_str))]
        elif filter_mode == 'exclude':
            if filter_logic == 'or':
                key_to_load = [k for k in key_to_load if not any((s in k for s in filter_str))]
            elif filter_logic == 'and':
                key_to_load = [k for k in key_to_load if not all((s in k for s in filter_str))]
    if filter_func is not None:
        key_to_load = [k for k in key_to_load if filter_func(k)]
    if ensure_config:
        if '_config' in metadata.keys() and '_config' not in key_to_load:
            key_to_load.append('_config')
    loaded_data = {}
    r = multi_process_list_for(process_num=process_num, func=part_load_dict_separate, for_list=os.listdir(load_dir), kwargs={'load_dir': load_dir, 'metadata': metadata, 'key_to_load': key_to_load}, func_name=f'load dict from {load_dir}', for_idx_name='subfile', use_tqdm=False)
    while r:
        k, v = r.pop(0)
        if k is not None:
            loaded_data[k] = v
    if set(loaded_data.keys()) != set(key_to_load):
        missing_keys = set(key_to_load) - set(loaded_data.keys())
        raise ValueError(f'Not all keys were loaded. Missing keys: {missing_keys}. Please check the files in {load_dir}.')
    return loaded_data

@_compatible_data_container
def load_dict_auto(filename, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    try:
        return load_dict_separate(filename, key_to_load=key_to_load, filter_str=filter_str, filter_mode=filter_mode, filter_logic=filter_logic, filter_func=filter_func, process_num=process_num, ensure_config=ensure_config)
    except:
        return load_pkl(os.path.join(filename))

@_compatible_data_container
def load_dict(*args, **kwargs):
    return load_dict_auto(*args, **kwargs)

def save_pkl(obj, filename, format='joblib', compress=0, protocol=None, add_ext='auto'):
    if format == 'joblib':
        save_joblib(obj, filename, compress=compress, protocol=protocol, add_ext=add_ext)
    elif format == 'pickle':
        save_pickle(obj, filename, add_ext=add_ext)

def load_pkl(filename):
    if filename.endswith('.joblib'):
        return load_joblib(filename, add_ext=False)
    elif filename.endswith('.pickle'):
        return load_pickle(filename, add_ext=False)
    elif filename.endswith('.pkl'):
        return load_pickle(filename, add_ext=False)
    else:
        for ext in ['.joblib', '.pickle', '.pkl']:
            if os.path.exists(filename + ext):
                return load_pkl(filename + ext)
        try:
            return load_joblib(filename, add_ext=False)
        except:
            try:
                return load_pickle(filename, add_ext=False)
            except:
                raise FileNotFoundError(f'File {filename} not found')

def save_pickle(obj, filename, add_ext='auto'):
    mkdir(os.path.dirname(filename))
    if add_ext == 'auto':
        if not filename.endswith('.pickle'):
            filename += '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename, add_ext='auto'):
    if add_ext == 'auto':
        ext_list = ['.pickle', '.pkl']
        detected_ext = False
        for ext in ext_list:
            if filename.endswith(ext):
                detected_ext = True
                break
        if not detected_ext:
            for ext in ext_list:
                if os.path.exists(filename + ext):
                    return load_pickle(filename + ext, add_ext=False)
            raise FileNotFoundError(f'File {filename} not found')
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_joblib(obj, filename, compress=0, protocol=None, add_ext='auto'):
    mkdir(os.path.dirname(filename))
    if add_ext == 'auto':
        if not filename.endswith('.joblib'):
            filename += '.joblib'
    joblib.dump(obj, filename, compress=compress, protocol=protocol)

def load_joblib(filename, add_ext='auto'):
    if add_ext == 'auto':
        if not filename.endswith('.joblib'):
            filename += '.joblib'
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File {filename} not found')
    obj = joblib.load(filename)
    return obj

def load_df(filename, index_col=0, index_dtype=str, col_dtype=str):
    if os.path.exists(filename):
        if filename.endswith('.pkl') or filename.endswith('.joblib') or filename.endswith('.pickle'):
            df = pd.read_pickle(filename)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filename, index_col=index_col, engine='openpyxl')
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename, index_col=index_col)
        df.index = df.index.astype(index_dtype)
        df.columns = df.columns.astype(col_dtype)
        df = df.apply(lambda x: x.astype(float) if x.dtype.kind in 'iufc' else x)
        return df
    else:
        for ext in ['.xlsx', '.csv', '.pkl', '.joblib', '.pickle']:
            if os.path.exists(filename + ext):
                return load_df(filename + ext, index_col, index_dtype, col_dtype)
        raise ValueError('File not found. Please check the file path and extension.')

def save_array(arr, filename):
    mkdir(os.path.dirname(filename))
    if not filename.endswith('.npy'):
        filename += '.npy'
    np.save(filename, arr)

def save_sps_array(arr, filename):
    mkdir(os.path.dirname(filename))
    if not filename.endswith('.npz'):
        filename += '.npz'
    sps.save_npz(filename, arr)

def load_array(filename):
    if not filename.endswith('.npy'):
        filename += '.npy'
    return np.load(filename, allow_pickle=True)

def load_sps_array(filename):
    if not filename.endswith('.npz'):
        filename += '.npz'
    return sps.load_npz(filename)

def save_txt(data, filename):
    mkdir(os.path.dirname(filename))
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'w') as f:
        f.write(str(data))

def load_txt(filename):
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

def check_if_multiprocessing():
    current_process = multiprocessing.current_process()
    if current_process.name == 'MainProcess':
        return False
    else:
        return True

def split_list(lst, n):
    length = len(lst)
    size = length // n
    remainder = length % n
    divided_list = []
    start = 0
    for i in range(n):
        sublist_size = size + 1 if i < remainder else size
        divided_list.append(lst[start:start + sublist_size])
        start += sublist_size
    return divided_list

def _check_process_num_func_args_kwargs(process_num, func_list, args_list, kwargs_list):
    len_set = {len(func_list), len(args_list), len(kwargs_list)}
    if len(len_set) > 1:
        raise ValueError('The length of func_list, args_list and kwargs_list must be the same.')
    if process_num != 1 and process_num != len(func_list):
        raise ValueError('The process_num must be equal to the length of func_list or 1.')

def basic_multi_process(process_num, func_list, args_list=None, kwargs_list=None, task_name=''):
    if args_list is None:
        args_list = [()] * len(func_list)
    if kwargs_list is None:
        kwargs_list = [{} for _ in func_list]
    _check_process_num_func_args_kwargs(process_num, func_list, args_list, kwargs_list)
    if process_num > 1 and check_if_multiprocessing():
        print_title('unable to use multiprocessing inside a multiprocessing process, use single process instead')
        process_num = 1
    if process_num != 1:
        print_title(f'Start {task_name} with {process_num} processes')
        results = []
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in zip(func_list, args_list, kwargs_list)]
            for future in futures:
                with FlexibleTry() as ft:
                    results.append(future.result())
                if not ft.success:
                    results.append(None)
        print_title(f'Finish {task_name}')
        return results
    elif process_num == 1:
        return [func(*args, **kwargs) for func, args, kwargs in zip(func_list, args_list, kwargs_list)]

def _prepare_for_multi_process(process_num, func_list, args_list, kwargs_list):
    if not isinstance(func_list, list):
        raise ValueError('func_list must be a list')
    if not isinstance(args_list, list) and args_list is not None:
        raise ValueError('args_list must be a list or None')
    if not isinstance(kwargs_list, list) and kwargs_list is not None:
        raise ValueError('kwargs_list must be a list or None')
    if args_list is None:
        args_list = [()]
    else:
        args_list = args_list.copy()
    if kwargs_list is None:
        kwargs_list = [{}]
    else:
        kwargs_list = kwargs_list.copy()
    max_length = max(len(func_list), len(args_list), len(kwargs_list))
    if len(func_list) != max_length:
        if len(func_list) == 1:
            func_list = func_list * max_length
        else:
            raise ValueError('The length of func_list must be equal to max_length or 1.')
    if len(args_list) != max_length:
        if len(args_list) == 1:
            args_list = args_list * max_length
        else:
            raise ValueError('The length of args_list must be equal to max_length or 1.')
    if len(kwargs_list) != max_length:
        if len(kwargs_list) == 1:
            kwargs_list = kwargs_list * max_length
        else:
            raise ValueError('The length of kwargs_list must be equal to max_length or 1.')
    for i, args in enumerate(args_list):
        if args is None:
            args_list[i] = ()
    for i, kwargs in enumerate(kwargs_list):
        if kwargs is None:
            kwargs_list[i] = {}
    return (process_num, func_list, args_list, kwargs_list)

def _run_multi_func(func_list, args_list, kwargs_list):
    results = []
    for func, args, kwargs in zip(func_list, args_list, kwargs_list):
        results.append(func(*args, **kwargs))
    return results

def multi_process_for_different_func(process_num, func_list, args_list=None, kwargs_list=None, task_name=''):
    process_num, func_list, args_list, kwargs_list = _prepare_for_multi_process(process_num, func_list, args_list, kwargs_list)
    divided_func_list = split_list(func_list, process_num)
    divided_args_list = split_list(args_list, process_num)
    divided_kwargs_list = split_list(kwargs_list, process_num)
    new_func_list = [_run_multi_func] * process_num
    new_args_list = [()] * process_num
    new_kwargs_list = [{'func_list': divided_func_list[i], 'args_list': divided_args_list[i], 'kwargs_list': divided_kwargs_list[i]} for i in range(process_num)]
    return flatten_list(basic_multi_process(process_num, func_list=new_func_list, args_list=new_args_list, kwargs_list=new_kwargs_list, task_name=task_name), level=1)

def multi_process(process_num, func, args_list=None, kwargs_list=None, func_name=''):

    def _get_input_l_len(l):
        if l is None:
            return 0
        elif isinstance(l, list):
            return len(l)
        else:
            raise ValueError('args_list and kwargs_list must be list or None')
    max_len = max(_get_input_l_len(args_list), _get_input_l_len(kwargs_list))
    if max_len <= 1:
        if process_num > 1:
            print_title(f'input function, args_list and kwargs_list are all not multiple, use {process_num} processes to repeat the function')
        func_list = [func] * process_num
    else:
        func_list = [func] * max_len
    return multi_process_for_different_func(process_num, func_list, args_list, kwargs_list, func_name)

def part_list_for(func, for_list, for_idx_name, use_tqdm, *args, **kwargs):
    if use_tqdm and len(for_list) != 1:
        iterable = tqdm.tqdm(for_list, total=len(for_list))
    else:
        iterable = for_list
    results = []
    for i in iterable:
        results.append(func(*args, **{**kwargs, for_idx_name: i}))
    return results

def multi_process_list_for(process_num, func, args=None, kwargs=None, for_list=None, for_idx_name='i', func_name='', use_tqdm=True):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if for_list is None:
        raise ValueError('for_list must not be None')
    for_list = list(for_list)
    if process_num > len(for_list):
        print_title(f'Reduce process_num from {process_num} to {len(for_list)} as process_num > len(for_list)')
        process_num = len(for_list)
    divided_list = split_list(for_list, process_num)
    args_list = [(func, divided, for_idx_name, False) + args for divided in divided_list]
    if use_tqdm:
        args_list[0] = (func, divided_list[0], for_idx_name, True) + args
    kwargs_list = [kwargs] * process_num
    return flatten_list(multi_process(process_num, part_list_for, args_list, kwargs_list, func_name), level=1)

def part_items_for(func, key_list, value_list, for_key_name, for_value_name, use_tqdm, *args, **kwargs):
    if use_tqdm and len(key_list) != 1:
        iterable = tqdm.tqdm(zip(key_list, value_list), total=len(key_list))
    else:
        iterable = zip(key_list, value_list)
    results = []
    for key, value in iterable:
        results.append(func(*args, **{**kwargs, for_key_name: key, for_value_name: value}))
    return results

def multi_process_items_for(process_num, func, args=None, kwargs=None, for_dict=None, for_key_name='key', for_value_name='value', func_name='', use_tqdm=True):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if process_num > len(for_dict):
        print_title(f'Reduce process_num from {process_num} to {len(for_dict)} as process_num > len(for_dict)')
        process_num = len(for_dict)
    divided_key_list = split_list(list(for_dict.keys()), process_num)
    divided_value_list = split_list(list(for_dict.values()), process_num)
    args_list = [(func, divided_key, divided_value, for_key_name, for_value_name, False) + args for divided_key, divided_value in zip(divided_key_list, divided_value_list)]
    if use_tqdm:
        args_list[0] = (func, divided_key_list[0], divided_value_list[0], for_key_name, for_value_name, True) + args
    kwargs_list = [kwargs] * process_num
    return flatten_list(multi_process(process_num, part_items_for, args_list, kwargs_list, func_name), level=1)

def get_tag(n, case=TAG_CASE, parentheses=TAG_PARENTHESES):
    if case not in ('lower', 'upper'):
        raise ValueError('Unknown case: ' + case)
    if n > 26:
        raise ValueError('n must not exceed 26')
    start = ord('a') if case == 'lower' else ord('A')
    tags = [chr(start + i) for i in range(n)]
    if parentheses:
        tags = ['(' + tag + ')' for tag in tags]
    return tags

def get_decimal_num(number, abs_tol=1e-05, rel_tol=1e-05):
    decimal_count = 0
    while True:
        rounded_number = round(number, decimal_count)
        abs_break = True
        rel_break = True
        if abs_tol is not None:
            if abs(rounded_number - number) > abs_tol:
                abs_break = False
        if rel_tol is not None:
            if abs(rounded_number - number) > rel_tol * abs(number):
                rel_break = False
        if abs_break and rel_break:
            break
        decimal_count += 1
    return decimal_count

def get_max_decimal_num(number_list, **kwargs):
    decimal_count_list = [get_decimal_num(number, **kwargs) for number in number_list]
    return max(decimal_count_list)

def format_text(text, text_process=None):
    text_process = update_dict(TEXT_PROCESS, text_process)
    capitalize = text_process['capitalize']
    replace_underscore = text_process['replace_underscore']
    ignore_equation_underscore = text_process['ignore_equation_underscore']
    if text is None:
        return None
    else:
        if ignore_equation_underscore:
            dollar_positions = []
            for index, char in enumerate(text):
                if char != '$':
                    continue
                backslash_count = 0
                previous = index - 1
                while previous >= 0 and text[previous] == '\\':
                    backslash_count += 1
                    previous -= 1
                if backslash_count % 2 == 0:
                    dollar_positions.append(index)
            if len(dollar_positions) % 2 == 0:
                dollar_positions = set(dollar_positions)
                in_math = False
                processed_chars = []
                for index, char in enumerate(text):
                    if index in dollar_positions:
                        in_math = not in_math
                    if char == '_' and (not in_math) and replace_underscore:
                        processed_chars.append(replace_underscore)
                    else:
                        processed_chars.append(char)
                parts = [''.join(processed_chars)]
            else:
                parts = [text.replace('_', replace_underscore) if replace_underscore else text]
        else:
            parts = [text.replace('_', replace_underscore) if replace_underscore else text]
        if capitalize:
            processed_parts = [part[0].capitalize() + part[1:] if part else '' for part in parts]
        else:
            processed_parts = parts
        return ''.join(processed_parts)

def sci_str_to_latex(sci_str):
    match = re.match('([+-]?\\d*\\.\\d+)e([+-]?\\d+)', sci_str)
    if match:
        base, exponent = match.groups()
        latex_str = f'${base} \\times 10^{{{exponent}}}$'
        return latex_str
    else:
        raise ValueError('The input string is not in scientific notation.')

def round_float(number, digits=ROUND_DIGITS, format_type=ROUND_FORMAT, latex=False):
    if format_type == 'general':
        result = f'{number:.{digits}g}'
        if 'e' in result:
            return round_float(number, digits=digits, format_type='scientific', latex=latex)
        else:
            return result
    elif format_type == 'standard':
        return f'{number:.{digits}f}'
    elif format_type == 'scientific':
        if latex:
            return sci_str_to_latex(f'{number:.{digits}e}')
        else:
            return f'{number:.{digits}e}'
    elif format_type == 'percent':
        return f'{number:.{digits}%}'

def get_coefficient_exponent(number, base=10):
    exponent = int(np.floor(np.log(abs(number)) / np.log(base)))
    coefficient = number / base ** exponent
    return (coefficient, exponent)

def sci_float_to_latex(number, base=10, base_str='10', round_digits=0):
    if number == 0:
        return '$0$'
    coefficient, exp = get_coefficient_exponent(number, base)
    if np.allclose(round_digits, 0):
        if np.allclose(coefficient, 1):
            return f'${base_str}^{{{exp}}}$'
        elif np.allclose(coefficient, -1):
            return f'$-{base_str}^{{{exp}}}$'
    return f'${coefficient:.{round_digits}f} \\times {base_str}^{{{exp}}}$'

def format_filename(filename, file_process=None):
    file_process = update_dict(FILENAME_PROCESS, file_process)
    if filename is None:
        return None
    else:
        return filename.replace(' ', file_process['replace_blank'])

def concat_str(strs, sep='_', rm_double_sep=True, ignore_none=True, ignore_empty=True, convert_non_str=True):
    if ignore_none:
        strs = [s for s in strs if s is not None]
    if ignore_empty:
        strs = [s for s in strs if s != '']
    if convert_non_str:
        strs = [str(s) for s in strs]
    result = sep.join(strs)
    if rm_double_sep:
        while sep * 2 in result:
            result = result.replace(sep * 2, sep)
    return result

def cat(*args, sep='_', rm_double_sep=True, ignore_none=True, ignore_empty=True, convert_non_str=True):
    return concat_str(strs=args, sep=sep, rm_double_sep=rm_double_sep, ignore_none=ignore_none, ignore_empty=ignore_empty, convert_non_str=convert_non_str)

def hash_or_str(key, sep='_', replace_dot=REPLACE_DOT):
    if isinstance(key, (str, int, float, bool)):
        if replace_dot:
            return str(key).replace('.', replace_dot)
        else:
            return str(key)
    if isinstance(key, tuple):
        if len(key) < 10:
            return str_tuple(key, sep=sep, replace_dot=replace_dot)
    return hashlib.md5(str(key).encode()).hexdigest()

def str_tuple(t, sep='_', replace_dot=REPLACE_DOT):
    result = sep.join(map(str, t))
    if replace_dot:
        result = result.replace('.', replace_dot)
    return result

def fill_missing_key(d, key_list, fill_value=None):
    warnings.warn('fill_missing_key,;fill_dict_kwargs.', FutureWarning, stacklevel=2)
    if d is None:
        return {k: fill_value for k in key_list}
    new_d = d.copy()
    for k in key_list:
        if k not in new_d:
            new_d[k] = fill_value
    return new_d

def create_dict(keys, values=None, default_value=None, deep_copy=True):
    if deep_copy:
        if values is None:
            return {key: copy.deepcopy(default_value) for key in keys}
        else:
            return {key: copy.deepcopy(value) for key, value in zip(keys, values)}
    elif values is None:
        return {key: default_value for key in keys}
    else:
        return {key: value for key, value in zip(keys, values)}

def filter_dict(d, keys):
    return {k: d[k] for k in keys if k in d}

def get_sub_dict(d, keys):
    return filter_dict(d, keys)

def update_dict(original_dict, new_dict):
    if original_dict is None:
        original_dict = {}
    if new_dict is None:
        return original_dict.copy()
    else:
        return {**original_dict, **new_dict}

def swap_dict_level(nested_dict, level_mapping):

    def _validate_level_mapping(level_mapping):
        for k, v in level_mapping.items():
            if v not in level_mapping.keys():
                raise ValueError(f'Invalid level mapping: {level_mapping}. Each new position must correspond to an existing original position.')

    def _flatten_dict(d, path=()):
        flat = {}
        for key, value in d.items():
            current_path = path + (key,)
            if isinstance(value, dict):
                flat.update(_flatten_dict(value, current_path))
            else:
                flat[current_path] = value
        return flat

    def _nest_dict(flat_dict):
        result = {}
        for path, value in flat_dict.items():
            current = result
            for key in path[:-1]:
                current = current.setdefault(key, {})
            current[path[-1]] = value
        return result
    _validate_level_mapping(level_mapping)
    flat_dict = _flatten_dict(nested_dict)
    if not flat_dict:
        return {}
    max_depth = len(next(iter(flat_dict.keys())))
    new_order = list(range(max_depth))
    for old_pos, new_pos in level_mapping.items():
        if old_pos < max_depth and new_pos < max_depth:
            new_order[new_pos] = old_pos
    rearranged_flat = {}
    for path, value in flat_dict.items():
        new_path = tuple((path[new_order[i]] for i in range(max_depth)))
        rearranged_flat[new_path] = value
    return _nest_dict(rearranged_flat)

class BuildKeyStrategy(abc.ABC):

    def __init__(self, included_name_list=None):
        if included_name_list is None:
            included_name_list = []
        self.included_name_list = included_name_list

    def validate_params(self, kwargs, allowed_params=None):
        if not kwargs:
            raise ValueError('Operation failed')
        if allowed_params is not None:
            invalid = [p for p in kwargs if p not in allowed_params]
            if invalid:
                raise ValueError(f':{invalid},:{allowed_params}')

    def _append_to_parts(self, parts, name, value):
        if name in self.included_name_list:
            parts.append(name)
        parts.append(value)

    @abc.abstractmethod
    def build_key(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        return self.build_key(**kwargs)

class SortedBuildKey(BuildKeyStrategy):

    def build_key(self, **kwargs):
        self.validate_params(kwargs)
        sorted_params = sorted(kwargs.keys())
        parts = []
        for name in sorted_params:
            value = kwargs[name]
            self._append_to_parts(parts, name, value)
        return cat(*parts)

class OrderedBuildKey(BuildKeyStrategy):

    def __init__(self, param_order, included_name_list=None):
        self.param_order = param_order
        super().__init__(included_name_list=included_name_list)

    def build_key(self, **kwargs):
        self.validate_params(kwargs, self.param_order)
        parts = []
        for param in self.param_order:
            if param in kwargs:
                value = kwargs[param]
                self._append_to_parts(parts, param, value)
        return cat(*parts)

class DataContainer(dict):

    def __init__(self, included_name_list=None):
        super().__init__()
        self.key_builder = SortedBuildKey(included_name_list=included_name_list)
        self['_config'] = {}
        self['_config']['included_name_list'] = self.key_builder.included_name_list
        self['_config']['param_map'] = {}
        self.ignore_key_list = ['_config']

    def build_key(self, **kwargs):
        return self.key_builder.build_key(**kwargs)

    def load(self):
        pass

class OrderedDataContainer(DataContainer):

    def __init__(self, param_order, included_name_list=None):
        super().__init__(included_name_list=included_name_list)
        self.key_builder = OrderedBuildKey(param_order=param_order, included_name_list=included_name_list)
        self['_config']['param_order'] = param_order

def _dict_to_data_container(d):
    config = d['_config'].copy()
    param_order = config.pop('param_order', None)
    param_map = {key: params.copy() for key, params in config.pop('param_map').items()}
    included_name_list = config.pop('included_name_list', None)
    if param_order is not None:
        param_order = param_order.copy()
    if included_name_list is not None:
        included_name_list = included_name_list.copy()
    if param_order is None:
        data_container = DataContainer(included_name_list=included_name_list)
    else:
        data_container = OrderedDataContainer(param_order=param_order, included_name_list=included_name_list)
    data_container['_config']['param_map'] = param_map
    for key, value in d.items():
        if key == '_config':
            continue
        data_container[key] = value
    return data_container

def flatten_list(input_list, level=None):
    if level is None:
        level = float('inf')

    def flatten_recursive(lst, curr_level):
        flattened = []
        for item in lst:
            if isinstance(item, list) and curr_level < level:
                flattened.extend(flatten_recursive(item, curr_level + 1))
            else:
                flattened.append(item)
        return flattened
    return flatten_recursive(input_list, 0)

def rebuild_list_with_index(flattened, original, index=0):
    result = []
    for item in original:
        if isinstance(item, list):
            sub_list, index = rebuild_list_with_index(flattened, item, index)
            result.append(sub_list)
        else:
            result.append(flattened[index])
            index += 1
    return (result, index)

def rebuild_list(flattened, original):
    return rebuild_list_with_index(flattened, original)[0]

def pure_list(l):
    if isinstance(l, list):
        return [pure_list(item) for item in l]
    elif isinstance(l, np.ndarray):
        if l.shape:
            return [pure_list(item) for item in l.tolist()]
        else:
            return l.item()
    elif isinstance(l, tuple):
        return [pure_list(item) for item in l]
    else:
        return l

def list_shape(lst):
    if not isinstance(lst, list):
        return ()
    if len(lst) == 0:
        return (0,)
    return (len(lst),) + list_shape(lst[0])

def reorder_df(df, new_row=None, new_col=None):
    if new_row is not None:
        df = df.loc[new_row]
    if new_col is not None:
        df = df[new_col]
    return df

def align_df_row_col_order(df, keep_which='row'):
    if keep_which == 'row':
        return df.sort_index(axis=1).sort_index(axis=0)
    elif keep_which == 'col':
        return df.sort_index(axis=0).sort_index(axis=1)

def get_sub_df(df, row_list=None, col_list=None):
    if row_list is not None:
        new_df = df.loc[row_list]
    else:
        new_df = df.copy()
    if col_list is not None:
        new_df = new_df[col_list]
    return new_df

def replace_df_diagonal(df, replace_value=0):
    df_copy = df.copy()
    for row in df_copy.index:
        for col in df_copy.columns:
            if row == col:
                df_copy.loc[row, col] = replace_value
    return df_copy

def get_zero_df(index, columns):
    df = pd.DataFrame(np.zeros((len(index), len(columns))), index=index, columns=columns)
    index_col_str(df)
    return df

def index_col_str(df):
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

def normalize_df_with_sum(df, row_or_col):
    if row_or_col == 'row':
        axis = 1
    elif row_or_col == 'col':
        axis = 0
    return df.div(df.sum(axis=axis), axis=1 - axis)

def row_sum_df(df):
    return df.sum(axis=1)

def col_sum_df(df):
    return df.sum(axis=0)

def insert_mid(arr):
    new_arr = []
    for i in range(len(arr)):
        new_arr.append(arr[i])
        if i < len(arr) - 1:
            new_arr.append((arr[i] + arr[i + 1]) / 2)
    if isinstance(arr, np.ndarray):
        return np.array(new_arr)
    else:
        return new_arr

def sort_dict(d, sort_by='value', reverse=False):
    if sort_by == 'key':
        return {k: d[k] for k in sorted(d.keys(), reverse=reverse)}
    elif sort_by == 'value':
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    else:
        raise ValueError("sort_by must be 'key' or 'value'")

def get_default_param(func):
    sig = inspect.signature(func)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

def get_param_value(func, args, kwargs, param_name):
    signature = inspect.signature(func)
    bound_args = signature.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments.get(param_name, None)

def process_inf(data, inf_policy=INF_POLICY, verbose=False):
    if inf_policy == 'to_nan':
        if isinstance(data, np.ndarray):
            inf_count = int(np.count_nonzero(np.isinf(data)))
            data = np.where(np.isinf(data), np.nan, data)
        elif isinstance(data, list):
            inf_count = int(np.count_nonzero(np.isinf(np.asarray(data))))
            data = np.where(np.isinf(data), np.nan, data).tolist()
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            inf_count = int(np.count_nonzero(np.asarray(data.isin([np.inf, -np.inf]))))
            data = data.replace([np.inf, -np.inf], np.nan)
        else:
            inf_count = 0
        if verbose and inf_count:
            print(f"process_inf: processed {inf_count} Inf value(s) with inf_policy='{inf_policy}'.")
    elif inf_policy == 'ignore':
        pass
    return data

def process_nan(data, nan_policy=NAN_POLICY, fill_value=None, verbose=False):
    if nan_policy == 'fill' and fill_value is None:
        raise ValueError("fill_value must be provided when nan_policy='fill'")
    nan_count_before = int(np.count_nonzero(np.asarray(pd.isna(data))))

    def df_like_interpolate(data):
        if isinstance(data, np.ndarray):
            return pd.Series(data).interpolate().values
        elif isinstance(data, list):
            return pd.Series(data).interpolate().tolist()
    if isinstance(data, list):
        data = np.array(data)
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = np.where(np.isnan(data), fill_value, data)
        elif nan_policy == 'drop':
            result = data[~np.isnan(data)]
        elif nan_policy == 'interpolate':
            result = df_like_interpolate(data)
        else:
            raise ValueError('nan_policy not supported')
        result = result.tolist()
    elif isinstance(data, np.ndarray):
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = np.where(np.isnan(data), fill_value, data)
        elif nan_policy == 'drop':
            if data.ndim > 1:
                raise ValueError('drop nan not supported for multi-dimensional arrays')
            result = data[~np.isnan(data)]
        elif nan_policy == 'interpolate':
            if data.ndim > 1:
                raise ValueError('interpolate nan not supported for multi-dimensional arrays')
            result = df_like_interpolate(data)
        else:
            raise ValueError('nan_policy not supported')
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        if nan_policy == 'propagate':
            result = data
        elif nan_policy == 'fill':
            result = data.fillna(fill_value)
        elif nan_policy == 'drop':
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    raise ValueError('drop nan not supported for multi-column DataFrames')
            result = data.dropna()
        elif nan_policy == 'interpolate':
            if isinstance(data, pd.DataFrame):
                if len(data.columns) > 1:
                    raise ValueError('interpolate nan not supported for multi-column DataFrames')
            result = data.interpolate()
        else:
            raise ValueError('nan_policy not supported')
    else:
        return None
    nan_count_after = int(np.count_nonzero(np.asarray(pd.isna(result))))
    processed_count = nan_count_before - nan_count_after
    if verbose and processed_count:
        print(f"process_nan: processed {processed_count} NaN value(s) with nan_policy='{nan_policy}'.")
    return result

def process_special_value(data, nan_policy=NAN_POLICY, fill_value=None, inf_policy=INF_POLICY, verbose=False):
    data = process_inf(data, inf_policy=inf_policy, verbose=verbose)
    data = process_nan(data, nan_policy=nan_policy, fill_value=fill_value, verbose=verbose)
    return data

def sync_special_value(*args, inf_policy=INF_POLICY, verbose=False):
    if not args:
        raise ValueError('At least one input is required')
    supported_types = (list, np.ndarray, pd.Series, pd.DataFrame)
    if not all((isinstance(a, supported_types) for a in args)):
        raise TypeError('All inputs must be lists, numpy arrays, pandas Series, or pandas DataFrames')
    input_type = type(args[0])
    if any((type(a) is not input_type for a in args[1:])):
        raise TypeError('All inputs must have the same type')
    shapes = [np.asarray(a).shape for a in args]
    if len(set(shapes)) > 1:
        raise ValueError('All inputs must have the same shape')
    if input_type is pd.Series:
        reference_index = args[0].index
        if any((not a.index.equals(reference_index) for a in args[1:])):
            raise ValueError('All pandas Series inputs must have the same index')
    elif input_type is pd.DataFrame:
        reference_index = args[0].index
        reference_columns = args[0].columns
        if any((not a.index.equals(reference_index) or not a.columns.equals(reference_columns) for a in args[1:])):
            raise ValueError('All pandas DataFrame inputs must have the same index and columns')
    args = [process_inf(a, inf_policy=inf_policy, verbose=verbose) for a in args]
    args_np = [np.array(a) if not isinstance(a, np.ndarray) else a for a in args]
    nan_mask = np.any(np.isnan(args_np), axis=0)
    synchronized_count = sum((int(np.count_nonzero(nan_mask & ~np.isnan(arg))) for arg in args_np))
    args_np_synced = [np.where(nan_mask, np.nan, arg) for arg in args_np]
    if verbose and synchronized_count:
        print(f'sync_special_value: synchronized {synchronized_count} value(s) to NaN.')
    args_synced = []
    for original_arg, synced_arg in zip(args, args_np_synced):
        if isinstance(original_arg, list):
            args_synced.append(synced_arg.tolist())
        elif isinstance(original_arg, pd.Series):
            args_synced.append(pd.Series(synced_arg, index=original_arg.index))
        elif isinstance(original_arg, pd.DataFrame):
            args_synced.append(pd.DataFrame(synced_arg, index=original_arg.index, columns=original_arg.columns))
        elif isinstance(original_arg, np.ndarray):
            args_synced.append(synced_arg)
    return args_synced if len(args_synced) > 1 else args_synced[0]

def has_invalid_value(value):
    if isinstance(value, dict):
        return any((has_invalid_value(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return any((has_invalid_value(item) for item in value))
    array = np.asarray(value)
    if array.dtype == object and array.ndim:
        return any((has_invalid_value(item) for item in array.flat))
    if np.asarray(pd.isna(value)).any():
        return True
    try:
        return np.isinf(array).any()
    except TypeError:
        return False

def _print_invalid_value_warning(func_name, **data):
    invalid_args = [name for name, value in data.items() if value is not None and has_invalid_value(value)]
    if invalid_args:
        print(f"{func_name}: {', '.join(invalid_args)} contains NaN or Inf.")

def scale_to_new_range(data, old_min, old_max, new_min, new_max):
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def lowess_smooth(x, y, frac=0.2):
    x = np.array(x)
    y = np.array(y)
    sort_index = np.argsort(x)
    x_sorted = x[sort_index]
    y_sorted = y[sort_index]
    lowess_result = lowess(y_sorted, x_sorted, frac=frac)
    x_smooth = lowess_result[:, 0]
    y_smooth = lowess_result[:, 1]
    return (x_smooth, y_smooth)

def get_angle(source, target, angle_type='rad', angle_range='0:360'):
    source = np.array(source)
    target = np.array(target)
    if len(source.shape) == 1:
        source = source.reshape(1, -1)
    if len(target.shape) == 1:
        target = target.reshape(1, -1)
    if source.shape[0] != target.shape[0] and source.shape[0] != 1 and (target.shape[0] != 1):
        raise ValueError('The number of source and target points must be the same.')
    angle = np.arctan2(target[:, 1] - source[:, 1], target[:, 0] - source[:, 0])
    if angle_type == 'deg':
        angle = np.degrees(angle)
    if angle_range == '0:360':
        if angle_type == 'deg':
            angle = np.where(angle < 0, angle + 360, angle)
        if angle_type == 'rad':
            angle = np.where(angle < 0, angle + 2 * np.pi, angle)
    if len(angle) == 1:
        angle = angle[0]
    return angle

def get_angle_3d(source, target, angle_type='rad', azim_range='0:360', elev_range='-90:90'):
    if angle_type not in ('rad', 'deg'):
        raise ValueError("angle_type must be 'rad' or 'deg'")
    if azim_range not in ('0:360', '-180:180'):
        raise ValueError("azim_range must be '0:360' or '-180:180'")
    if elev_range not in ('0:180', '-90:90'):
        raise ValueError("elev_range must be '0:180' or '-90:90'")
    source = np.array(source)
    target = np.array(target)
    if len(source.shape) == 1:
        source = source.reshape(1, -1)
    if len(target.shape) == 1:
        target = target.reshape(1, -1)
    if source.shape[0] != target.shape[0] and source.shape[0] != 1 and (target.shape[0] != 1):
        raise ValueError('The number of source and target points must be the same.')
    dx = target[:, 0] - source[:, 0]
    dy = target[:, 1] - source[:, 1]
    dz = target[:, 2] - source[:, 2]
    azim = np.arctan2(dy, dx)
    elev = np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2))
    if angle_type == 'deg':
        azim = np.degrees(azim)
        elev = np.degrees(elev)
    if azim_range == '0:360':
        if angle_type == 'deg':
            azim = np.where(azim < 0, azim + 360, azim)
        else:
            azim = np.where(azim < 0, azim + 2 * np.pi, azim)
    if elev_range == '0:180':
        if angle_type == 'deg':
            elev = 90 - elev
        else:
            elev = np.pi / 2 - elev
    if len(azim) == 1:
        azim = azim[0]
        elev = elev[0]
    return (azim, elev)

def get_corr(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, sync=True, verbose=False):
    x, y = sync_special_value(x, y, inf_policy=inf_policy, verbose=verbose) if sync else (x, y)
    return np.corrcoef(process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose), process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose))[0, 1]

def get_rectangle_matrix_corr(matrix_x, matrix_y, diagonal=True, get_corr_kwargs=None):
    if get_corr_kwargs is None:
        get_corr_kwargs = {}
    if diagonal:
        matrix_x_flat = matrix_x.flatten()
        matrix_y_flat = matrix_y.flatten()
    else:
        rows, cols = matrix_x.shape
        mask = np.eye(rows, cols, dtype=bool)
        non_diag_mask = ~mask
        matrix_x_flat = matrix_x[non_diag_mask].flatten()
        matrix_y_flat = matrix_y[non_diag_mask].flatten()
    return get_corr(matrix_x_flat, matrix_y_flat, **get_corr_kwargs)

def get_pd_rectangle_corr(df_x, df_y, diagonal=True, get_corr_kwargs=None):
    df_x_aligned = align_df_row_col_order(df_x)
    df_y_aligned = align_df_row_col_order(df_y)
    return get_rectangle_matrix_corr(df_x_aligned.values, df_y_aligned.values, diagonal=diagonal, get_corr_kwargs=get_corr_kwargs)

def get_linregress(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, sync=True, verbose=False):
    x, y = sync_special_value(x, y, inf_policy=inf_policy, verbose=verbose) if sync else (x, y)
    x = np.array(process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose))
    y = np.array(process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose))
    regress_dict = {}
    regress_dict['slope'], regress_dict['intercept'], regress_dict['r'], regress_dict['p'], regress_dict['std_err'] = st.linregress(x, y)
    regress_dict['residual'] = y - (x * regress_dict['slope'] + regress_dict['intercept'])
    return regress_dict

def get_ks(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False):
    return get_ks_result(x, y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)['ks']

def get_ks_result(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False):
    x_clean = process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    if callable(y):
        ks_result = st.kstest(x_clean, y)
    else:
        y_clean = process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
        ks_result = st.ks_2samp(x_clean, y_clean)
    r = {'ks': ks_result.statistic, 'p': ks_result.pvalue}
    return r

def get_kl_divergence(p, q, base=None, **kwargs):
    return st.entropy(p, q, base=base, **kwargs)

def get_jsd(p, q, base=None, **kwargs):
    m = (np.array(p) + np.array(q)) / 2
    return (get_kl_divergence(p, m, base=base, **kwargs) + get_kl_divergence(q, m, base=base, **kwargs)) / 2

def get_hist(data, bins, stat='probability', nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False):
    local_data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    hist, bin_edges = np.histogram(local_data, bins=bins)
    bin_widths = np.diff(bin_edges)
    if stat == 'count':
        values = hist
    elif stat == 'frequency':
        values = hist / bin_widths
    elif stat == 'probability' or stat == 'proportion':
        values = hist / hist.sum()
    elif stat == 'percent':
        values = hist / hist.sum() * 100
    elif stat == 'density':
        values = hist / (hist.sum() * bin_widths)
    else:
        raise ValueError('Unsupported stat method.')
    return (values, bin_edges, get_midpoint(bin_edges))

def get_hist_2d(data_x, data_y, bins_x, bins_y, stat='probability', nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, sync=True, verbose=False):
    if sync:
        local_data_x, local_data_y = sync_special_value(data_x, data_y, inf_policy=inf_policy, verbose=verbose)
    else:
        local_data_x, local_data_y = (data_x.copy(), data_y.copy())
    local_data_x = process_special_value(local_data_x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    local_data_y = process_special_value(local_data_y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    hist, bin_edges_x, bin_edges_y = np.histogram2d(local_data_x, local_data_y, bins=[bins_x, bins_y])
    bin_widths_x = np.diff(bin_edges_x)
    bin_widths_y = np.diff(bin_edges_y)
    bin_areas = np.outer(bin_widths_x, bin_widths_y)
    if stat == 'count':
        values = hist
    elif stat == 'frequency':
        values = hist / bin_areas
    elif stat == 'probability' or stat == 'proportion':
        values = hist / hist.sum()
    elif stat == 'percent':
        values = hist / hist.sum() * 100
    elif stat == 'density':
        values = hist / (hist.sum() * bin_areas)
    else:
        raise ValueError('Unsupported stat method.')
    return (values, bin_edges_x, bin_edges_y, get_midpoint(bin_edges_x), get_midpoint(bin_edges_y))

def get_cdf(data, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False, **kwargs):
    clean_data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    return ECDF(clean_data, **kwargs)

def get_midpoint(x):
    return (x[1:] + x[:-1]) / 2

def repeat_data(data, repeat_times):
    if isinstance(data, list):
        if isinstance(repeat_times, int):
            return [item for item in data for _ in range(repeat_times)]
        elif isinstance(repeat_times, (list, np.ndarray)) and len(data) == len(repeat_times):
            return [item for item, count in zip(data, repeat_times) for _ in range(count)]
    elif isinstance(data, np.ndarray):
        if isinstance(repeat_times, (list, np.ndarray, int)):
            return np.repeat(data, repeat_times)
    if isinstance(data, dict) and isinstance(repeat_times, (dict, int)):
        result = []
        if isinstance(repeat_times, int):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend([item for item in value for _ in range(repeat_times)])
                else:
                    result.extend([value for _ in range(repeat_times)])
        elif isinstance(repeat_times, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend([item for item in value for _ in range(repeat_times[key])])
                else:
                    result.extend([value for _ in range(repeat_times[key])])
        return result
    if isinstance(data, pd.Series):
        if isinstance(repeat_times, int):
            return data.repeat(repeat_times).tolist()
        elif isinstance(repeat_times, pd.Series):
            repeated_values = [pd.Series([value] * repeat_times.get(index, 1)) for index, value in data.items()]
            repeated_series = pd.concat(repeated_values).reset_index(drop=True)
            return repeated_series.tolist()
    if isinstance(data, pd.DataFrame):
        if isinstance(repeat_times, int):
            return data.loc[data.index.repeat(repeat_times)].to_numpy()
        elif isinstance(repeat_times, pd.DataFrame):
            data_flat = data.to_numpy().flatten()
            repeats_flat = repeat_times.to_numpy().flatten().astype(int)
            return np.repeat(data_flat, repeats_flat)

def get_iterable_ax(ax):
    if isinstance(ax, np.ndarray):
        return ax.flatten()
    elif isinstance(ax, list):
        return flatten_list(pure_list(ax))
    elif isinstance(ax, dict):
        return list(ax.values())
    else:
        return [ax]

def get_iterable_ax_for_decorator(ax):
    if isinstance(ax, np.ndarray):
        return (ax.flatten(), True)
    elif isinstance(ax, list):
        return (flatten_list(pure_list(ax)), True)
    elif isinstance(ax, dict):
        return (list(ax.values()), True)
    else:
        return (ax, False)

def iterate_over_obj(func, obj_name):

    @wraps(func)
    def wrapper(*args, **kwargs):
        obj = get_param_value(func, args, kwargs, obj_name)
        iterable_obj, is_iterable = get_iterable_ax_for_decorator(obj)
        if is_iterable:
            results = []
            signature = inspect.signature(func)
            for item in iterable_obj:
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                bound_args.arguments[obj_name] = item
                result = func(*bound_args.args, **bound_args.kwargs)
                results.append(result)
            return rebuild_ax(results, obj)
        else:
            return func(*args, **kwargs)
    return wrapper

def iterate_over_axs(func):
    return iterate_over_obj(func, 'ax')

def rebuild_ax(flatten_ax, original_ax):
    if isinstance(original_ax, np.ndarray):
        result = np.empty(original_ax.shape, dtype=object)
        if original_ax.ndim == 1:
            for i in range(len(flatten_ax)):
                result[i] = flatten_ax[i]
        elif original_ax.ndim == 2:
            for i in range(original_ax.shape[0]):
                for j in range(original_ax.shape[1]):
                    result[i, j] = flatten_ax[i * original_ax.shape[1] + j]
        return result
    elif isinstance(original_ax, list):
        return rebuild_list(flatten_ax, original_ax)
    elif isinstance(original_ax, dict):
        return dict(zip(original_ax.keys(), flatten_ax))
    else:
        return flatten_ax

def squeeze_ax(ax):
    if isinstance(ax, np.ndarray):
        if ax.size == 1:
            if ax.ndim == 2:
                return ax[0, 0]
            elif ax.ndim == 1:
                return ax[0]
        else:
            return np.squeeze(ax)
    else:
        return ax

def filter_data_by_xylim(x, y, xlim=None, ylim=None):
    if xlim or ylim:
        if xlim is not None:
            if xlim[0] is None:
                xlim = (-np.inf, xlim[1])
            if xlim[1] is None:
                xlim = (xlim[0], np.inf)
        if ylim is not None:
            if ylim[0] is None:
                ylim = (-np.inf, ylim[1])
            if ylim[1] is None:
                ylim = (ylim[0], np.inf)
        filtered_data = [(xi, yi) for xi, yi in zip(x, y) if (xlim is None or xlim[0] <= xi <= xlim[1]) and (ylim is None or ylim[0] <= yi <= ylim[1])]
        if filtered_data:
            x_filtered, y_filtered = zip(*filtered_data)
            return (list(x_filtered), list(y_filtered))
        else:
            return ([], [])
    return (x.copy(), y.copy())

def plt_scatter(ax, x, y, label=None, color=BLUE, vert=True, rasterized=False, rasterized_threshold=10000, xlim=None, ylim=None, linewidths=0.0, **kwargs):
    _print_invalid_value_warning('plt_scatter', x=x, y=y)
    x, y = filter_data_by_xylim(x, y, xlim=xlim, ylim=ylim)
    if not vert:
        x, y = (y, x)
    if rasterized_threshold is not None and rasterized_threshold is not False:
        if isinstance(x, np.ndarray) or isinstance(x, list):
            if len(x) > rasterized_threshold:
                rasterized = True
    if 'c' in kwargs:
        local_kwargs = kwargs.copy()
        local_kwargs['c'] = pure_list(local_kwargs['c'])
        if len(list_shape(local_kwargs['c'])) == 1:
            local_kwargs['c'] = [local_kwargs['c']]
        return ax.scatter(x, y, label=label, rasterized=rasterized, linewidths=linewidths, **local_kwargs)
    else:
        return ax.scatter(x, y, label=label, rasterized=rasterized, color=color, linewidths=linewidths, **kwargs)

def plt_line(ax, x, y, label=None, color=BLUE, vert=True, xlim=None, ylim=None, **kwargs):
    _print_invalid_value_warning('plt_line', x=x, y=y)
    x, y = filter_data_by_xylim(x, y, xlim=xlim, ylim=ylim)
    if not vert:
        x, y = (y, x)
    return ax.plot(x, y, label=label, color=color, **kwargs)

def plt_bar(ax, x, y, label=None, color=BLUE, vert=True, equal_space=False, err=None, capsize=PLT_CAP_SIZE, ecolor=BLACK, elabel=None, width=BAR_WIDTH, **kwargs):
    _print_invalid_value_warning('plt_bar', x=x, y=y, err=err)
    if equal_space:
        x = [str(i) for i in x]
    error_kw = kwargs.pop('error_kw', None)
    if err is not None:
        if error_kw is None:
            error_kw = {}
        if 'elinewidth' not in error_kw:
            error_kw['elinewidth'] = ERRORBAR_LINEWIDTH
        if 'capthick' not in error_kw:
            error_kw['capthick'] = ERRORBAR_LINEWIDTH
        add_errorbar(ax, x, y, err, vert=vert, label=elabel, color=ecolor, capsize=capsize, **kwargs)
    if vert:
        return ax.bar(x, y, label=label, color=color, yerr=err, capsize=capsize, ecolor=ecolor, width=width, error_kw=error_kw, **kwargs)
    else:
        return ax.barh(x, y, label=label, color=color, xerr=err, capsize=capsize, ecolor=ecolor, height=width, error_kw=error_kw, **kwargs)

def plt_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
    plt_hist_instance = PltHist(ax, data, bins=bins, label=label, color=color, stat=stat, vert=vert, **kwargs)
    return plt_hist_instance.bar

class PltHist:

    def __init__(self, ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
        _print_invalid_value_warning('plt_hist', data=data)
        self.hist, self.edge, self.mid_point = get_hist(data, bins, stat=stat)
        set_ax(ax, ylabel=stat)
        self.bar = plt_bar(ax, self.mid_point, self.hist, label=label, color=color, width=self.edge[1] - self.edge[0], vert=vert, **kwargs)

def plt_polygon(ax, xy, color=BLUE, fill=True, adjust_lim=True, **kwargs):
    _print_invalid_value_warning('plt_polygon', xy=xy)
    polygon = plt.Polygon(xy, color=color, fill=fill, **kwargs)
    if adjust_lim:
        for sub_xy in xy:
            ax.scatter(sub_xy[0], sub_xy[1], s=0, zorder=-1)
    return ax.add_patch(polygon)

def plt_bar_dict(ax, data, label=None, color=BLUE, width=BAR_WIDTH, vert=True, **kwargs):
    x = range(len(data))
    y = list(data.values())
    set_ax(ax, xtick=x, xtick_label=list(data.keys()))
    return plt_bar(ax, x, y, label=label, color=color, width=width, vert=vert, **kwargs)

def sns_heatmap(ax, data, cmap=HEATMAP_CMAP, square=True, cbar=True, cbar_position=None, cbar_label=None, discrete_label=None, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, show_xtick=True, show_ytick=True, show_all_xtick=True, show_all_ytick=True, xtick_fontsize=None, ytick_fontsize=None, mask=None, mask_color=MASK_COLOR, mask_tick='mask', norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, heatmap_kwargs=None, cbar_kwargs=None):
    text_process = update_dict(TEXT_PROCESS, text_process)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    if xtick_fontsize is None:
        xtick_fontsize = TICK_SIZE
    if ytick_fontsize is None:
        ytick_fontsize = TICK_SIZE
    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()
    if isinstance(mask, np.ndarray):
        mask = pd.DataFrame(mask)
    elif mask is not None:
        mask = mask.copy()
    if mask is not None:
        local_data = local_data.mask(mask)
    if vmin is None:
        vmin = np.nanmin(local_data.values)
    elif vmin > np.nanmin(local_data.values):
        cbar_kwargs['add_leq'] = True
    if vmax is None:
        vmax = np.nanmax(local_data.values)
    elif vmax < np.nanmax(local_data.values):
        cbar_kwargs['add_geq'] = True
    if heatmap_kwargs is None:
        local_heatmap_kwargs = {}
    else:
        local_heatmap_kwargs = heatmap_kwargs.copy()
        if 'annot' in local_heatmap_kwargs:
            annot_kws = dict(local_heatmap_kwargs.get('annot_kws') or {})
            if 'fontsize' not in annot_kws:
                ax_width, ax_height = get_ax_size(ax)
                from_height = suitable_tick_size(data.shape[0], ax_height)
                from_width = suitable_tick_size(data.shape[1], ax_width) / 3
                annot_kws['fontsize'] = min(from_height, from_width)
                local_heatmap_kwargs['annot_kws'] = annot_kws
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    norm = get_norm(vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs)
    if np.allclose(vmin, vmax):
        local_cmap = get_cmap([cmap(0.5), cmap(0.5)])
        sns.heatmap(local_data, ax=ax, cmap=local_cmap, square=square, cbar=False, norm=norm, **local_heatmap_kwargs)
    else:
        sns.heatmap(local_data, ax=ax, cmap=cmap, square=square, cbar=False, norm=norm, **local_heatmap_kwargs)
    if mask is not None and mask_color is not None:
        use_mask = True
        mask_cmap = get_cmap([mask_color, mask_color])
        sns.heatmap(mask, ax=ax, cmap=mask_cmap, cbar=False, mask=~mask, square=square)
    else:
        use_mask = False
    if show_xtick:
        if show_all_xtick:
            ax.set_xticks(np.arange(len(local_data.columns)) + 0.5)
            ax.set_xticklabels(local_data.columns, rotation=xtick_rotation, fontsize=xtick_fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation, fontsize=xtick_fontsize)
        ax.xaxis.set_tick_params(labelbottom=show_xtick)
    else:
        ax.set_xticks([])
    if show_ytick:
        if show_all_ytick:
            ax.set_yticks(np.arange(len(local_data.index)) + 0.5)
            ax.set_yticklabels(local_data.index, rotation=ytick_rotation, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation, fontsize=ytick_fontsize)
        ax.yaxis.set_tick_params(labelleft=show_ytick)
    else:
        ax.set_yticks([])
    if cbar:
        cbars = add_side_colorbar(ax, mappable=None, cmap=cmap, cbar_position=cbar_position, cbar_label=cbar_label, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, discrete_label=discrete_label, use_mask=use_mask, mask_color=mask_color, mask_tick=mask_tick, **cbar_kwargs)
        return cbars

def add_errorbar(ax, x, y, err, label=None, color=BLACK, linestyle='None', capsize=PLT_CAP_SIZE, vert=True, equal_space=False, **kwargs):
    _print_invalid_value_warning('add_errorbar', x=x, y=y, err=err)
    if 'elinewidth' not in kwargs:
        kwargs['elinewidth'] = ERRORBAR_LINEWIDTH
    if isinstance(x, (list, tuple, np.ndarray)):
        x = list(x)
        if any((isinstance(item, str) for item in x)) or equal_space:
            x = [str(item) for item in x]
            if vert:
                ax.set_xticks(np.arange(len(x)))
                ax.set_xticklabels(x)
            else:
                ax.set_yticks(np.arange(len(x)))
                ax.set_yticklabels(x)
            x = np.arange(len(x))
    if vert:
        return ax.errorbar(x, y, yerr=err, label=label, color=color, linestyle=linestyle, capsize=capsize, **kwargs)
    else:
        return ax.errorbar(y, x, xerr=err, label=label, color=color, linestyle=linestyle, capsize=capsize, **kwargs)

@iterate_over_axs
def inset_ax(ax, left, right, bottom, top, label='inset', inset_mode='fig', **kwargs):
    valid_inset_modes = ('fig', 'ax')
    if inset_mode not in valid_inset_modes:
        raise ValueError(f'inset_mode must be one of {valid_inset_modes}, got {inset_mode!r}.')
    if inset_mode == 'fig':
        ax_left, ax_bottom, ax_width, ax_height = ax.get_position().bounds
        new_left = ax_left + left * ax_width
        new_right = ax_left + right * ax_width
        new_bottom = ax_bottom + bottom * ax_height
        new_top = ax_bottom + top * ax_height
        return add_ax(ax.get_figure(), new_left, new_right, new_bottom, new_top, label=cat(ax.get_label(), label), **kwargs)
    elif inset_mode == 'ax':
        width = right - left
        height = top - bottom
        new_ax = ax.inset_axes([left, bottom, width, height], label=cat(ax.get_label(), label), **kwargs)
        add_custom_child_to_ax(ax, new_ax, label)
        return new_ax

def add_ax(fig, left, right, bottom, top, label='add', **kwargs):
    return fig.add_axes([left, bottom, right - left, top - bottom], label=label, **kwargs)

@iterate_over_axs
def add_side_ax(ax, position='right', relative_size=SIDE_PAD * 2, pad=SIDE_PAD, sharex=None, sharey=None, hide_repeat_xaxis=False, hide_repeat_yaxis=False, label='side', inset_mode='fig', spine_mode='same', **kwargs):
    valid_positions = ('right', 'left', 'top', 'bottom')
    if position not in valid_positions:
        raise ValueError(f'position must be one of {valid_positions}, got {position!r}.')
    if sharex is True:
        sharex = ax
    elif sharex is None or sharex is False:
        sharex = None
    if sharey is True:
        sharey = ax
    elif sharey is None or sharey is False:
        sharey = None
    new_pos = {'left': 0.0, 'right': 1.0, 'top': 1.0, 'bottom': 0.0}
    if position == 'right':
        new_pos['left'] = 1.0 + pad
        new_pos['right'] = 1.0 + pad + relative_size
    elif position == 'left':
        new_pos['left'] = -relative_size - pad
        new_pos['right'] = -pad
    elif position == 'top':
        new_pos['top'] = 1.0 + pad + relative_size
        new_pos['bottom'] = 1.0 + pad
    elif position == 'bottom':
        new_pos['top'] = -pad
        new_pos['bottom'] = -relative_size - pad
    new_ax = inset_ax(ax, left=new_pos['left'], right=new_pos['right'], bottom=new_pos['bottom'], top=new_pos['top'], sharex=sharex, sharey=sharey, label=label, inset_mode=inset_mode, **kwargs)
    if spine_mode == 'same':
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(ax.spines[location].get_visible())
    elif spine_mode == 'all':
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(True)
    elif spine_mode in ['none', 'hide'] or not spine_mode:
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(False)
    if sharex is not None and hide_repeat_xaxis:
        if position == 'top':
            new_ax.xaxis.set_visible(False)
        elif position == 'bottom':
            new_ax.xaxis.set_visible(False)
    if sharey is not None and hide_repeat_yaxis:
        if position == 'right':
            new_ax.yaxis.set_visible(False)
        elif position == 'left':
            new_ax.yaxis.set_visible(False)
    return new_ax

@iterate_over_axs
def rm_ax(ax):
    ax.remove()

@iterate_over_axs
def set_ax_view_3d(ax, elev=ELEV, azim=AZIM):
    if isinstance(ax, Axes3D):
        ax.view_init(elev=elev, azim=azim)

def get_ax_position_custom(ax):
    left, bottom, width, height = ax.get_position().bounds
    right = left + width
    top = bottom + height
    return (left, right, bottom, top)

def set_ax_position_custom(ax, left, right, bottom, top):
    ax.set_position([left, bottom, right - left, top - bottom])

def set_relative_ax_position(ax, nrows=1, ncols=1, margin=None, squeeze=False):
    if margin is None:
        margin = MARGIN.copy()
    subplot_width = 1 / ncols
    subplot_height = 1 / nrows
    ax_width = (margin['right'] - margin['left']) * subplot_width
    ax_height = (margin['top'] - margin['bottom']) * subplot_height
    if nrows > 1 and ncols > 1:
        for row in range(nrows):
            for col in range(ncols):
                left = margin['left'] / ncols + col * subplot_width
                bottom = margin['bottom'] / nrows + (nrows - row - 1) * subplot_height
                ax[row, col].set_position([left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols > 1:
        for col in range(ncols):
            left = margin['left'] / ncols + col * subplot_width
            bottom = margin['bottom']
            if squeeze:
                ax[col].set_position([left, bottom, ax_width, ax_height])
            else:
                ax[0, col].set_position([left, bottom, ax_width, ax_height])
    if nrows > 1 and ncols == 1:
        for row in range(nrows):
            left = margin['left']
            bottom = margin['bottom'] / nrows + (nrows - row - 1) * subplot_height
            if squeeze:
                ax[row].set_position([left, bottom, ax_width, ax_height])
            else:
                ax[row, 0].set_position([left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols == 1:
        left = margin['left']
        bottom = margin['bottom']
        if squeeze:
            ax.set_position([left, bottom, ax_width, ax_height])
        else:
            ax[0, 0].set_position([left, bottom, ax_width, ax_height])
    return ax

@iterate_over_axs
def align_ax(ax, ref_ax, align_mode='horizontal', keep_size=False):
    if align_mode == 'horizontal' or align_mode == 'vertical' or align_mode == 'all':
        if keep_size:
            print('may not be able to keep size when align_mode is horizontal or vertical')
    ref_pos_left, ref_pos_right, ref_pos_bottom, ref_pos_top = get_ax_position_custom(ref_ax)
    pos_left, pos_right, pos_bottom, pos_top = get_ax_position_custom(ax)
    pos_width = pos_right - pos_left
    pos_height = pos_top - pos_bottom
    if align_mode == 'horizontal':
        pos_top = ref_pos_top
        pos_bottom = ref_pos_bottom
    if align_mode == 'vertical':
        pos_left = ref_pos_left
        pos_right = ref_pos_right
    if align_mode == 'all':
        pos_left = ref_pos_left
        pos_right = ref_pos_right
        pos_bottom = ref_pos_bottom
        pos_top = ref_pos_top
    if align_mode == 'left':
        pos_left = ref_pos_left
        if keep_size:
            pos_right = pos_left + pos_width
    if align_mode == 'right':
        pos_right = ref_pos_right
        if keep_size:
            pos_left = pos_right - pos_width
    if align_mode == 'top':
        pos_top = ref_pos_top
        if keep_size:
            pos_bottom = pos_top - pos_height
    if align_mode == 'bottom':
        pos_bottom = ref_pos_bottom
        if keep_size:
            pos_top = pos_bottom + pos_height
    set_ax_position_custom(ax, pos_left, pos_right, pos_bottom, pos_top)

@iterate_over_axs
def move_ax(ax, d_left=None, d_right=None, d_bottom=None, d_top=None, keep_size=False):
    left, right, bottom, top = get_ax_position_custom(ax)
    if keep_size:
        position_dict = {'left': left, 'right': right, 'bottom': bottom, 'top': top}
        if d_left is not None:
            position_dict['left'] -= d_left
            position_dict['right'] -= d_left
        if d_right is not None:
            position_dict['left'] += d_right
            position_dict['right'] += d_right
        if d_bottom is not None:
            position_dict['bottom'] -= d_bottom
            position_dict['top'] -= d_bottom
        if d_top is not None:
            position_dict['bottom'] += d_top
            position_dict['top'] += d_top
        left, right, bottom, top = (position_dict['left'], position_dict['right'], position_dict['bottom'], position_dict['top'])
    else:
        if d_left is not None:
            left -= d_left
        if d_right is not None:
            right += d_right
        if d_bottom is not None:
            bottom -= d_bottom
        if d_top is not None:
            top += d_top
    set_ax_position_custom(ax, left, right, bottom, top)

def calculate_gs_coordinates(ncols, nrows, left, right, bottom, top, wspace, hspace, width_ratios, height_ratios):
    if width_ratios is None:
        width_ratios = [1] * ncols
    if height_ratios is None:
        height_ratios = [1] * nrows
    if wspace is None:
        wspace = plt.rcParams['figure.subplot.wspace']
    if hspace is None:
        hspace = plt.rcParams['figure.subplot.hspace']
    width_total = right - left
    height_total = top - bottom
    width_ratios_sum = sum(width_ratios)
    height_ratios_sum = sum(height_ratios)
    cell_widths = [r / width_ratios_sum * (width_total - (len(width_ratios) - 1) * wspace) for r in width_ratios]
    cell_heights = [r / height_ratios_sum * (height_total - (len(height_ratios) - 1) * hspace) for r in height_ratios]
    x_coords = [left]
    y_coords = [top]
    for width in cell_widths:
        x_coords.append(x_coords[-1] + width + wspace)
    for height in cell_heights:
        y_coords.append(y_coords[-1] - height - hspace)
    coordinates = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            cell_left = x_coords[j]
            cell_right = x_coords[j + 1] - wspace
            cell_bottom = y_coords[i + 1] + hspace
            cell_top = y_coords[i]
            coordinates[i, j] = (cell_left, cell_right, cell_bottom, cell_top)
    return coordinates

@iterate_over_axs
def split_ax_by_gs(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None, sharex=False, sharey=False, squeeze=True, keep_original=False, label='split', inset_mode='fig', **kwargs):
    if inset_mode == 'fig':
        gs = get_gs_inside_ax(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
        label = cat(label, ax.get_label())
        sub_ax = get_all_ax_from_gs(gs, squeeze=squeeze, label=label, **kwargs)
        if keep_original is True:
            pass
        elif keep_original is False:
            rm_ax(ax)
            sub_ax = share_axis(sub_ax, sharex=sharex, sharey=sharey)
        else:
            if sharex in ['row', 'col'] or sharey in ['row', 'col']:
                print('sharex or sharey can not be row or col in keep_original mode')
            sub_ax = share_axis_to_target(sub_ax, ax, sharex=sharex, sharey=sharey)
            align_ax(ax, sub_ax[keep_original], align_mode='all')
            rm_ax(sub_ax[keep_original])
            sub_ax[keep_original] = ax
    elif inset_mode == 'ax':
        coords = calculate_gs_coordinates(ncols=ncols, nrows=nrows, left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
        sub_ax = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                local_label = cat(label, f'row{i}_col{j}')
                sub_ax[i, j] = inset_ax(ax, coords[i][j][0], coords[i][j][1], coords[i][j][2], coords[i][j][3], label=local_label, inset_mode='ax', **kwargs)
        if keep_original is True:
            pass
        elif keep_original is False:
            rm_ax_spine(ax)
            rm_ax_tick(ax)
            rm_ax_ticklabel(ax)
            sub_ax = share_axis(sub_ax, sharex=sharex, sharey=sharey)
        else:
            print('unsupport keep_original in inset_mode=ax')
        if squeeze:
            sub_ax = squeeze_ax(sub_ax)
    return sub_ax

def split_with_double_marginal_ax(ax, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, label='marginal', inset_mode='fig'):
    if x_side_ax_position == 'top':
        height_ratios = [x_side_ax_size, 1 - x_side_ax_size]
        x_side_index = 0
    elif x_side_ax_position == 'bottom':
        height_ratios = [1 - x_side_ax_size, x_side_ax_size]
        x_side_index = 1
    if y_side_ax_position == 'right':
        width_ratios = [1 - y_side_ax_size, y_side_ax_size]
        y_side_index = 1
    elif y_side_ax_position == 'left':
        width_ratios = [y_side_ax_size, 1 - y_side_ax_size]
        y_side_index = 0
    if inset_mode == 'fig':
        keep_original = (1 - x_side_index, 1 - y_side_index)
        sub_ax = split_ax_by_gs(ax, nrows=2, ncols=2, hspace=x_side_ax_pad, wspace=y_side_ax_pad, height_ratios=height_ratios, width_ratios=width_ratios, keep_original=keep_original, label=label, inset_mode=inset_mode)
    elif inset_mode == 'ax':
        print_title('inset_mode=ax need to delete original ax and re-create an ax in the needed position, thus not recommended, please insure that other process on ax is done after this function')
        sub_ax = split_ax_by_gs(ax, nrows=2, ncols=2, hspace=x_side_ax_pad, wspace=y_side_ax_pad, height_ratios=height_ratios, width_ratios=width_ratios, keep_original=False, label=label, inset_mode=inset_mode)
    x_side_ax = sub_ax[x_side_index, 1 - y_side_index]
    y_side_ax = sub_ax[1 - x_side_index, y_side_index]
    ax = sub_ax[1 - x_side_index, 1 - y_side_index]
    x_side_ax = share_axis_to_target(x_side_ax, ax, sharex=True, sharey=False)
    y_side_ax = share_axis_to_target(y_side_ax, ax, sharex=False, sharey=True)
    rm_ax(sub_ax[x_side_index, y_side_index])
    return (x_side_ax, y_side_ax, ax)

def merge_ax(axs, rm_mode='rm_axis', label='merge'):
    axs = get_iterable_ax(axs)
    for ax in axs:
        if rm_mode == 'rm_axis':
            rm_ax_axis(ax)
        elif rm_mode == 'rm_ax':
            rm_ax(ax)
        else:
            pass
        label = cat(label, ax.get_label())
    left = get_extreme_ax_position(axs, 'left')
    right = get_extreme_ax_position(axs, 'right')
    bottom = get_extreme_ax_position(axs, 'bottom')
    top = get_extreme_ax_position(axs, 'top')
    return add_ax(fig=axs[0].get_figure(), left=left, right=right, bottom=bottom, top=top, label=label)

def add_custom_child_to_ax(ax, child, child_name):
    if not hasattr(ax, 'custom_children'):
        ax.custom_children = {}
    if child_name not in ax.custom_children:
        ax.custom_children[child_name] = child
    else:
        new_child_name = child_name + '_new'
        add_custom_child_to_ax(ax, child, new_child_name)

def get_extreme_ax_position(axs, position):
    axs = get_iterable_ax(axs)
    if position not in ['left', 'right', 'top', 'bottom']:
        raise ValueError('position')
    if position == 'left':
        return min([ax.get_position().x0 for ax in axs])
    elif position == 'right':
        return max([ax.get_position().x1 for ax in axs])
    elif position == 'top':
        return max([ax.get_position().y1 for ax in axs])
    elif position == 'bottom':
        return min([ax.get_position().y0 for ax in axs])

def get_linear_but_log_formatter(base='10', label_format='auto'):

    def linear_but_log_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(x)
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return '${}^{{{}}}$'.format(base, local_label_format % x)
    return FuncFormatter(linear_but_log_formatter)

@iterate_over_axs
def set_linear_but_log_axis(ax, axis=None, base='10', label_format='auto'):
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_xticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.xaxis.set_major_formatter(get_linear_but_log_formatter(base=base, label_format=local_label_format))
        if axi == 'y':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_yticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.yaxis.set_major_formatter(get_linear_but_log_formatter(base=base, label_format=local_label_format))

def get_log_e_formatter(label_format='auto'):

    def log_e_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(np.log(x))
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return '$e^{{{}}}$'.format(local_label_format % np.log(x))
    return FuncFormatter(log_e_formatter)

@iterate_over_axs
def set_log_e_axis(ax, axis=None, label_format='auto'):
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(np.log(ax.get_xticks()))
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.xaxis.set_major_formatter(get_log_e_formatter(label_format=local_label_format))
        if axi == 'y':
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(np.log(ax.get_yticks()))
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.yaxis.set_major_formatter(get_log_e_formatter(label_format=local_label_format))

def get_sym_positive_formatter(label_format='auto'):

    def sym_positive_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(x)
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return local_label_format % abs(x)
    return FuncFormatter(sym_positive_formatter)

@iterate_over_axs
def set_sym_positive_axis(ax, axis=None, bound=None, label_format='auto'):
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            try:
                if bound is None:
                    bound = max(abs(ax.get_xlim()))
                ax.set_xlim(-bound, bound)
            except:
                pass
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_xticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.xaxis.set_major_formatter(get_sym_positive_formatter(label_format=local_label_format))
        if axi == 'y':
            try:
                if bound is None:
                    bound = max(abs(ax.get_ylim()))
                ax.set_ylim(-bound, bound)
            except:
                pass
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_yticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format
            ax.yaxis.set_major_formatter(get_sym_positive_formatter(label_format=local_label_format))

def get_norm(norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    if norm_mode == 'linear':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = 0
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 1
            else:
                vmax = local_norm_kwargs.pop('vmax')
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'log':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = 0.1
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 1
            else:
                vmax = local_norm_kwargs.pop('vmax')
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'symlog':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = -10
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 10
            else:
                vmax = local_norm_kwargs.pop('vmax')
        if 'linthresh' not in local_norm_kwargs:
            linthresh = 1
        else:
            linthresh = local_norm_kwargs.pop('linthresh')
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'two_slope':
        local_norm_kwargs = norm_kwargs.copy()
        if vmin is None:
            if 'vmin' not in local_norm_kwargs:
                vmin = -1
            else:
                vmin = local_norm_kwargs.pop('vmin')
        if vmax is None:
            if 'vmax' not in local_norm_kwargs:
                vmax = 1
            else:
                vmax = local_norm_kwargs.pop('vmax')
        if 'vcenter' not in local_norm_kwargs:
            vcenter = 0
        else:
            vcenter = local_norm_kwargs.pop('vcenter')
        norm = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax, **local_norm_kwargs)
    elif norm_mode == 'boundary':
        local_norm_kwargs = norm_kwargs.copy()
        if 'boundaries' not in local_norm_kwargs:
            local_norm_kwargs['boundaries'] = np.linspace(0, 1, 6, endpoint=True)
        if 'ncolors' not in local_norm_kwargs:
            local_norm_kwargs['ncolors'] = 256
        norm = mcolors.BoundaryNorm(**local_norm_kwargs)
    return norm

def get_cmap(colors, continuous=True):
    if len(colors) == 1:
        local_colors = colors * 2
    else:
        local_colors = colors.copy()
    if continuous:
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', local_colors)
    else:
        cmap = mcolors.ListedColormap(colors)
    return cmap

@iterate_over_axs
def add_colorbar(ax, mappable=None, cmap=CMAP, ticks=None, tick_labels=None, discrete_label=None, display_edge_ticks=True, cbar_position=None, cbar_label=None, use_mask=False, mask_color=MASK_COLOR, mask_pos='start', mask_pad=0, mask_cbar_ratio=None, mask_tick='mask', mask_tick_loc=None, label_size=None, tick_size=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, inset_mode='fig'):
    norm_kwargs = update_dict({}, norm_kwargs)
    if mappable is None:
        norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
    if label_kwargs is None:
        label_kwargs = {}
    if label_size is None:
        label_size = CBAR_LABEL_SIZE
    if tick_size is None:
        tick_size = CBAR_TICK_SIZE
    if vmin is None:
        vmin = mappable.norm.vmin
    if vmax is None:
        vmax = mappable.norm.vmax
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    text_process = update_dict(TEXT_PROCESS, text_process)
    mappable.set_clim(vmin, vmax)
    if vmin < mappable.norm.vmin:
        add_leq = True
    if vmax > mappable.norm.vmax:
        add_geq = True
    if mask_cbar_ratio is None:
        mask_cbar_ratio = 0.2
        if norm_mode == 'boundary':
            mask_cbar_ratio = 1 / mappable.norm.boundaries.size
    cbar_label = format_text(cbar_label, text_process=text_process)
    mask_tick = format_text(mask_tick, text_process=text_process)
    if discrete_label is not None:
        discrete_label = [format_text(label, text_process=text_process) for label in discrete_label]
    if cbar_position['position'] in ['right', 'left']:
        orientation = 'vertical'
    if cbar_position['position'] in ['top', 'bottom']:
        orientation = 'horizontal'
    if use_mask:
        if cbar_position['position'] in ['right', 'left']:
            if mask_pos == 'start':
                cbar_ax, mask_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=mask_pad, height_ratios=[1 - mask_cbar_ratio, mask_cbar_ratio], label='cbar_mask', inset_mode=inset_mode)
            if mask_pos == 'end':
                mask_ax, cbar_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=mask_pad, height_ratios=[mask_cbar_ratio, 1 - mask_cbar_ratio], label='mask_cbar', inset_mode=inset_mode)
        if cbar_position['position'] in ['top', 'bottom']:
            if mask_pos == 'start':
                mask_ax, cbar_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=mask_pad, width_ratios=[mask_cbar_ratio, 1 - mask_cbar_ratio], label='mask_cbar', inset_mode=inset_mode)
            if mask_pos == 'end':
                cbar_ax, mask_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=mask_pad, width_ratios=[1 - mask_cbar_ratio, mask_cbar_ratio], label='cbar_mask', inset_mode=inset_mode)
        ax.axis('off')
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=get_cmap([mask_color, mask_color]))
        mask_cbar = plt.colorbar(sm, cax=mask_ax, orientation=orientation)
        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation=orientation)
    else:
        cbar = plt.colorbar(mappable, cax=ax, orientation=orientation)
    if use_mask:
        if norm_mode == 'boundary' and discrete_label is None:
            mask_cbar.set_ticks([0.0])
        else:
            mask_cbar.set_ticks([0.5])
        if mask_tick_loc is not None:
            mask_cbar.set_ticks([mask_tick_loc])
    cbar.set_label(cbar_label, fontsize=label_size, **label_kwargs)
    if norm_mode == 'boundary':
        if 'boundaries' not in norm_kwargs:
            raise ValueError('norm_modeboundary,norm_kwargsboundaries')
        if 'ncolors' not in norm_kwargs:
            raise ValueError('norm_modeboundary,norm_kwargsncolors')
    if norm_mode == 'boundary':
        if ticks is None:
            ticks = mappable.norm.boundaries
            if discrete_label is not None:
                ticks = insert_mid(ticks)
                if not display_edge_ticks:
                    ticks = ticks[1::2]
    else:
        if ticks is None:
            ticks = cbar.get_ticks()
        ticks = [tick for tick in ticks if vmin <= tick <= vmax]
        if add_leq:
            if not np.allclose(vmin, ticks[0]):
                ticks = [vmin] + ticks
        if add_geq:
            if not np.allclose(vmax, ticks[-1]):
                ticks = ticks + [vmax]
    cbar.set_ticks(ticks)
    if tick_labels is None:
        if formatter is None:
            if norm_mode == 'linear' or norm_mode == 'two_slope' or norm_mode == 'boundary':
                tick_labels = [round_float(tick, round_digits, round_format_type, latex=True) for tick in ticks]
            elif norm_mode == 'log' or norm_mode == 'symlog':
                tick_labels = [sci_float_to_latex(tick) for tick in ticks]
        else:
            if formatter_kwargs is None:
                formatter_kwargs = {}
            tick_labels = [formatter(tick, **formatter_kwargs) for tick in ticks]
    if norm_mode == 'boundary':
        if discrete_label is not None:
            if display_edge_ticks:
                for i in range(len(ticks) - 1):
                    if i % 2 == 1:
                        tick_labels[i] = discrete_label[i // 2]
            else:
                tick_labels = discrete_label.copy()
    if adjust_tick_size:
        num_ticks = len(cbar.get_ticks())
        width, height = get_ax_size(cbar.ax)
        if cbar_position['position'] in ['right', 'left']:
            plt_size = height
        elif cbar_position['position'] in ['top', 'bottom']:
            plt_size = width
            tick_size = suitable_tick_size(num_ticks, plt_size, tick_size, tick_proportion)
    if add_leq:
        if np.allclose(vmin, ticks[0]):
            tick_labels[0] = '≤' + tick_labels[0]
    if add_geq:
        if np.allclose(vmax, ticks[-1]):
            tick_labels[-1] = '≥' + tick_labels[-1]
    cbar.set_ticklabels(tick_labels, fontsize=tick_size)
    if use_mask:
        mask_cbar.set_ticklabels([mask_tick], fontsize=tick_size)
    if use_mask:
        cbar_list = [cbar, mask_cbar]
    else:
        cbar_list = [cbar]
    for cb in cbar_list:
        if cbar_position['position'] == 'right':
            pass
        elif cbar_position['position'] == 'left':
            cb.ax.yaxis.set_ticks_position('left')
        elif cbar_position['position'] == 'top':
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.set_xticks(cb.ax.get_xticks())
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        elif cbar_position['position'] == 'bottom':
            cb.ax.xaxis.set_ticks_position('bottom')
            cb.ax.set_xticks(cb.ax.get_xticks())
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        cb.outline.set_visible(False)
        cb.ax.xaxis.set_tick_params(which='minor', length=0)
        cb.ax.yaxis.set_tick_params(which='minor', length=0)
    if use_mask:
        add_custom_child_to_ax(ax, mask_cbar, f'mask_cbar_{mask_tick}')
        add_custom_child_to_ax(ax, cbar, f'cbar_{cbar_label}')
        return [cbar, mask_cbar]
    else:
        add_custom_child_to_ax(ax, cbar, f'cbar_{cbar_label}')
        return [cbar]

@iterate_over_axs
def add_side_colorbar(ax, mappable=None, cmap=CMAP, ticks=None, tick_labels=None, discrete_label=None, display_edge_ticks=True, cbar_position=None, cbar_label=None, use_mask=False, mask_color=MASK_COLOR, mask_pos='start', mask_pad=0, mask_cbar_ratio=None, mask_tick='mask', mask_tick_loc=None, label_size=None, tick_size=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, inset_mode='fig'):
    if isinstance(ax, Axes3D):
        cbar_position = update_dict(CBAR_POSITION_3D, cbar_position)
    else:
        cbar_position = update_dict(CBAR_POSITION, cbar_position)
    side_ax = add_side_ax(ax, cbar_position['position'], cbar_position['size'], cbar_position['pad'], inset_mode=inset_mode)
    return add_colorbar(side_ax, mappable=mappable, cmap=cmap, ticks=ticks, tick_labels=tick_labels, discrete_label=discrete_label, display_edge_ticks=display_edge_ticks, cbar_position=cbar_position, cbar_label=cbar_label, use_mask=use_mask, mask_color=mask_color, mask_pos=mask_pos, mask_pad=mask_pad, mask_cbar_ratio=mask_cbar_ratio, mask_tick=mask_tick, mask_tick_loc=mask_tick_loc, label_size=label_size, tick_size=tick_size, adjust_tick_size=adjust_tick_size, tick_proportion=tick_proportion, label_kwargs=label_kwargs, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, text_process=text_process, formatter=formatter, formatter_kwargs=formatter_kwargs, round_digits=round_digits, round_format_type=round_format_type, add_leq=add_leq, add_geq=add_geq, inset_mode=inset_mode)

def add_vline(ax, x, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=None, **kwargs):
    _print_invalid_value_warning('add_vline', x=x)
    if linewidth is None:
        linewidth = LINE_WIDTH
    return ax.axvline(x, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

def add_hline(ax, y, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=None, **kwargs):
    _print_invalid_value_warning('add_hline', y=y)
    if linewidth is None:
        linewidth = LINE_WIDTH
    return ax.axhline(y, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

def add_sline(ax, slope, intercept, reference_point='auto', label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=None, **kwargs):
    _print_invalid_value_warning('add_sline', slope=slope, intercept=intercept, reference_point=reference_point)
    if linewidth is None:
        linewidth = LINE_WIDTH
    if reference_point == 'auto':
        reference_point = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    reference_intercept = slope * reference_point + intercept
    return ax.axline((reference_point, reference_intercept), slope=slope, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

def add_patch(ax, patch, auto_scale=True):
    ax.add_patch(patch)
    if auto_scale:
        ax.autoscale_view()

def add_circle_patch(ax, center, radius, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=None, **kwargs):
    _print_invalid_value_warning('add_circle_patch', center=center, radius=radius)
    if lw is None:
        lw = LINE_WIDTH
    p = mpatches.Circle(center, radius, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p

def add_arc_patch(ax, center, radius, theta1, theta2, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=None, angle_type='rad', **kwargs):
    _print_invalid_value_warning('add_arc_patch', center=center, radius=radius, theta1=theta1, theta2=theta2)
    if angle_type == 'rad':
        theta1 = np.rad2deg(theta1)
        theta2 = np.rad2deg(theta2)
    elif angle_type == 'deg':
        pass
    else:
        raise ValueError("angle_type must be 'rad' or 'deg'")
    if lw is None:
        lw = LINE_WIDTH
    p = mpatches.Arc(center, 2 * radius, 2 * radius, theta1=theta1, theta2=theta2, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p

def add_arrow(ax, x_start, y_start, x_end, y_end, xycoords='data', label=None, fc=RED, ec=RED, linewidth=None, arrow_sytle=ARROW_STYLE, head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH, alpha=1.0, **kwargs):
    _print_invalid_value_warning('add_arrow', x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end)
    if linewidth is None:
        linewidth = LINE_WIDTH
    x_start, y_start, x_end, y_end = map(np.atleast_1d, [x_start, y_start, x_end, y_end])
    n_arrows = len(x_start)
    for i in range(n_arrows):
        add_annotation(ax, '', (x_end[i], y_end[i]), (x_start[i], y_start[i]), xycoords=xycoords, arrowprops={'fc': fc, 'ec': ec, 'linewidth': linewidth, 'arrowstyle': arrow_sytle, 'head_length': head_length, 'head_width': head_width, 'alpha': alpha}, **kwargs)
    ax.arrow([], [], [], [], label=label, fc=fc, ec=ec, linewidth=linewidth, head_width=0, head_length=0, **kwargs)

def add_annotation(ax, text, xy, xytext, xycoords='data', fontsize=None, arrowprops=None, text_process=None, **kwargs):
    text_process = update_dict(TEXT_PROCESS, text_process)
    arrowprops = update_dict(ARROW_PROPS, arrowprops)
    _print_invalid_value_warning('add_annotation', xy=xy, xytext=xytext)
    if fontsize is None:
        fontsize = FONT_SIZE
    if 'head_length' in arrowprops.keys() and 'head_length' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_length=' + str(arrowprops['head_length'])
        del arrowprops['head_length']
    if 'head_width' in arrowprops.keys() and 'head_width' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_width=' + str(arrowprops['head_width'])
        del arrowprops['head_width']
    if xycoords == 'data':
        ax.scatter([xy[0], xytext[0]], [xy[1], xytext[1]], s=0)
    text = format_text(text, text_process)
    return ax.annotate(text, xy, xytext=xytext, xycoords=xycoords, fontsize=fontsize, arrowprops=arrowprops, **kwargs)

def _check_maybe_wrong_transform_for_text(x, y, transform):
    if transform != 'data':
        if x > 2 or x < -2 or y > 2 or (y < -2):
            print(f'Warning: x={x}, y={y}, transform={transform} may be wrong.')

def add_text(ax, text, x=TEXT_X, y=TEXT_Y, text_process=None, transform='ax', va=TEXT_VA, ha=TEXT_HA, fontsize=None, color=BLACK, **kwargs):
    text_process = update_dict(TEXT_PROCESS, text_process)
    _print_invalid_value_warning('add_text', x=x, y=y)
    if fontsize is None:
        fontsize = FONT_SIZE
    _check_maybe_wrong_transform_for_text(x, y, transform)
    text = format_text(text, text_process)
    if transform == 'ax':
        transform = ax.transAxes
    elif transform == 'fig':
        transform = ax.figure.transFigure
    elif transform == 'data':
        transform = ax.transData
    return ax.text(x, y, text, transform=transform, va=va, ha=ha, fontsize=fontsize, color=color, **kwargs)

def add_multi_text_by_dict(ax, x_dict, y_dict, text_process=None, transform='ax', va=TEXT_VA, ha=TEXT_HA, fontsize=None, color=BLACK, **kwargs):
    if set(x_dict.keys()) != set(y_dict.keys()):
        raise ValueError('The keys of x_dict and y_dict must be the same.')
    text_obj_list = []
    for key in x_dict.keys():
        text_obj_list.append(add_text(ax, key, x=x_dict[key], y=y_dict[key], text_process=text_process, transform=transform, va=va, ha=ha, fontsize=fontsize, color=color, **kwargs))
    return text_obj_list

def auto_adjust_text(ax, text_obj_list, add_arrow, seed, arrow_style='->', arrow_color=BLACK, arrow_width=0.5, expand_points=(1.2, 1.4), expand_text=(1.2, 1.4), force_points=(0.5, 0.5), force_text=(0.5, 0.5), min_arrow_len=15, ensure_inside_axes=False, **kwargs):
    if add_arrow:
        arrow_props = dict(arrowstyle=arrow_style, color=arrow_color, lw=arrow_width)
    else:
        arrow_props = None
    adjust_text_custom.adjust_text(text_obj_list, arrowprops=arrow_props, expand_points=expand_points, expand_text=expand_text, force_points=force_points, force_text=force_text, min_arrow_len=min_arrow_len, ax=ax, ensure_inside_axes=ensure_inside_axes, seed=seed, **kwargs)

def add_ax_tag(ax, tag, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    return add_text(ax, tag, x=x, y=y, fontsize=fontsize, va=va, ha=ha, **kwargs)

def add_ax_tag_by_ref_ax(ax, tag, ref_ax, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    if x > 0:
        raise ValueError('x should be less than 0')
    if y < 1:
        raise ValueError('y should be greater than 1')
    if ref_ax is not None:
        ref_ax_width, ref_ax_height = get_ax_size(ref_ax)
        ax_width, ax_height = get_ax_size(ax)
        x = x * ref_ax_width / ax_width
        y = (y - 1) * ref_ax_height / ax_height + 1
    return add_ax_tag(ax, tag, x=x, y=y, fontsize=fontsize, va=va, ha=ha, **kwargs)

def add_axes_dict_tag(axes_dict, tag_dict=None, ref_ax=None, **kwargs):
    if tag_dict is None:
        tag_dict = {ax_name: ax_name for ax_name in axes_dict.keys()}
    for ax_name, tag in tag_dict.items():
        add_ax_tag_by_ref_ax(axes_dict[ax_name], tag, ref_ax, **kwargs)

def add_axes_list_tag_by_order(axes_list, tag_kwargs=None, ref_ax=None, **kwargs):
    tag_kwargs = update_dict({}, tag_kwargs)
    tag_list = get_tag(len(axes_list), **tag_kwargs)
    for i, ax in enumerate(axes_list):
        add_ax_tag_by_ref_ax(ax, tag_list[i], ref_ax, **kwargs)

def plt_edge_scatter(ax, x, y, color=BLUE, edge_color=BLACK, s=None, edge_prop=1.1, label=None, density=False, edge_kwargs=None, sc_kwargs=None):
    if edge_kwargs is None:
        edge_kwargs = {}
    if sc_kwargs is None:
        sc_kwargs = {}
    if s is None:
        s = MARKER_SIZE ** 2
    edge_size = s * edge_prop
    plt_scatter(ax, x, y, color=edge_color, s=edge_size, **edge_kwargs)
    if density:
        local_sc_kwargs = sc_kwargs.copy()
        if 'scatter_kwargs' in sc_kwargs:
            local_sc_kwargs['scatter_kwargs'] = update_dict(sc_kwargs['scatter_kwargs'], {'s': s})
        else:
            local_sc_kwargs = update_dict(sc_kwargs, {'scatter_kwargs': {'s': s}})
        return plt_density_scatter(ax, x, y, label=label, **local_sc_kwargs)
    else:
        local_sc_kwargs = update_dict(sc_kwargs, {'s': s})
        return plt_scatter(ax, x, y, color=color, label=label, **local_sc_kwargs)

def plt_colorful_scatter(ax, x, y, c, cmap=CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, s=None, label=None, label_cmap_float=1.0, scatter_kwargs=None, cbar=True, cbar_postion=None, cbar_kwargs=None):
    _print_invalid_value_warning('plt_colorful_scatter', c=c)
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if s is None:
        s = MARKER_SIZE ** 2
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)
    norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)
    local_scatter_kwargs = update_dict(scatter_kwargs, {'s': s})
    sc = plt_scatter(ax, x, y, color=None, c=c, cmap=cmap, label=None, norm=norm, **local_scatter_kwargs)
    if isinstance(label_cmap_float, int):
        print('warning: "label_cmap_float" is an integer, so it will be transformed to float')
    ax.scatter([], [], color=cmap(float(label_cmap_float)), label=label, s=s)
    if cbar:
        if vmin > np.nanmin(c):
            cbar_kwargs['add_leq'] = True
        if vmax < np.nanmax(c):
            cbar_kwargs['add_geq'] = True
        cbars = add_side_colorbar(ax, sc, vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs, cmap=cmap, cbar_position=cbar_postion, **cbar_kwargs)
        return (sc, cbars)
    else:
        return sc

def plt_group_bar(ax, x, y, label_list, width=None, inner_gap=0.0, colors=CMAP, vert=True, **kwargs):
    _print_invalid_value_warning('plt_group_bar', x=x)
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(label_list)))
    num_groups = len(y)
    num_bars = len(y[0])
    if width is None:
        width = BAR_WIDTH / (num_bars + (num_bars - 1) * inner_gap)
    gap_width = width * inner_gap
    indices = np.arange(num_groups)
    for i, lbl in enumerate(label_list):
        offset = (i - (num_bars - 1) / 2) * (width + gap_width)
        plt_bar(ax, indices + offset, [y[j][i] for j in range(num_groups)], label=lbl, color=colors[i], width=width, vert=vert, **kwargs)
    if vert:
        ax.set_xticks(indices)
        ax.set_xticklabels(x)
    else:
        ax.set_yticks(indices)
        ax.set_yticklabels(x)

def plt_two_side_bar(ax, x, y1, y2, label1=None, label2=None, width=BAR_WIDTH, color1=BLUE, color2=RED, yerr1=None, yerr2=None, capsize=PLT_CAP_SIZE, ecolor1=BLACK, ecolor2=BLACK, elabel1=None, elabel2=None, vert=True, equal_space=False, **kwargs):
    plt_bar(ax, x, np.array(y1), label=label1, color=color1, vert=vert, equal_space=equal_space, err=yerr1, capsize=capsize, ecolor=ecolor1, elabel=elabel1, width=width, **kwargs)
    plt_bar(ax, x, -np.array(y2), label=label2, color=color2, vert=vert, equal_space=equal_space, err=yerr2, capsize=capsize, ecolor=ecolor2, elabel=elabel2, width=width, **kwargs)
    local_max = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) * 1.2
    if vert:
        axis = 'y'
    else:
        axis = 'x'
    set_sym_positive_axis(ax, axis, local_max)

def plt_linregress(ax, x, y, xlog=False, ylog=False, xlog_base=10, ylog_base=10, xlog_base_str=None, ylog_base_str=None, linear_but_log=False, label=None, scatter_color=BLUE, line_color=RED, line_bound='data', show_list=None, round_digit_dict=None, round_format_dict=None, text_size=None, scatter_kwargs=None, line_kwargs=None, text_kwargs=None, regress_kwargs=None, show_scatter=True, scatter_mode='original'):
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    if regress_kwargs is None:
        regress_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    if show_list is None:
        show_list = ['r', 'p']
    if text_size is None:
        text_size = FONT_SIZE
    _print_invalid_value_warning('plt_linregress', x=x, y=y)
    x = np.array(x)
    y = np.array(y)
    if xlog:
        x_local = np.log(x) / np.log(xlog_base)
    else:
        x_local = x.copy()
    if ylog:
        y_local = np.log(y) / np.log(ylog_base)
    else:
        y_local = y.copy()
    if show_scatter:
        if linear_but_log:
            if scatter_mode == 'density':
                plt_density_scatter(ax, x_local, y_local, label=label, **scatter_kwargs)
            elif scatter_mode == 'edge':
                plt_edge_scatter(ax, x_local, y_local, color=scatter_color, label=label, **scatter_kwargs)
            else:
                plt_scatter(ax, x_local, y_local, color=scatter_color, label=label, **scatter_kwargs)
        elif scatter_mode == 'density':
            plt_density_scatter(ax, x, y, label=label, **scatter_kwargs)
        elif scatter_mode == 'edge':
            plt_edge_scatter(ax, x, y, color=scatter_color, label=label, **scatter_kwargs)
        else:
            plt_scatter(ax, x, y, color=scatter_color, label=label, **scatter_kwargs)
    if xlog:
        if linear_but_log:
            if xlog_base_str is None:
                if np.allclose(xlog_base, np.e):
                    xlog_base_str = 'e'
                else:
                    xlog_base_str = str(xlog_base)
            set_linear_but_log_axis(ax, 'x', xlog_base_str)
        else:
            ax.set_xscale('log', base=xlog_base)
            if np.allclose(xlog_base, np.e):
                set_log_e_axis(ax, 'x')
    if ylog:
        if linear_but_log:
            if ylog_base_str is None:
                if np.allclose(ylog_base, np.e):
                    ylog_base_str = 'e'
                else:
                    ylog_base_str = str(ylog_base)
            set_linear_but_log_axis(ax, 'y', ylog_base_str)
        else:
            ax.set_yscale('log', base=ylog_base)
            if np.allclose(ylog_base, np.e):
                set_log_e_axis(ax, 'y')
    if x.size > 1 and y.size > 1:
        regress_dict = get_linregress(x_local, y_local, **regress_kwargs)
        if line_bound == 'ax':
            if xlog or ylog:
                raise ValueError('xlog and ylog cannot be True when line_bound is "ax"')
            add_sline(ax, regress_dict['slope'], regress_dict['intercept'], reference_point='auto', color=line_color, **line_kwargs)
        elif line_bound == 'data':
            x_local_min, x_local_max = (np.nanmin(x_local), np.nanmax(x_local))
            y_local_min = x_local_min * regress_dict['slope'] + regress_dict['intercept']
            y_local_max = x_local_max * regress_dict['slope'] + regress_dict['intercept']
            if xlog and (not linear_but_log):
                x_min, x_max = (xlog_base ** x_local_min, xlog_base ** x_local_max)
            else:
                x_min, x_max = (x_local_min, x_local_max)
            if ylog and (not linear_but_log):
                y_min, y_max = (ylog_base ** y_local_min, ylog_base ** y_local_max)
            else:
                y_min, y_max = (y_local_min, y_local_max)
            plt_line(ax, [x_min, x_max], [y_min, y_max], color=line_color, **line_kwargs)
        add_text_by_dict(ax, text_dict=regress_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=text_size, **text_kwargs)
        return regress_dict

def plt_density_scatter(ax, x, y, label=None, label_cmap_float=1.0, estimate_type='kde', bw_method='auto', bins_x=BIN_NUM, bins_y=BIN_NUM, cmap=DENSITY_CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cbar=True, cbar_label='density', cbar_position=None, scatter_kwargs=None, cbar_kwargs=None):
    _print_invalid_value_warning('plt_density_scatter', x=x, y=y)
    if scatter_kwargs is None:
        scatter_kwargs = {}
    cbar_kwargs = update_dict(cbar_kwargs, {'cbar_label': cbar_label})
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    x_array = np.array(x)
    y_array = np.array(y)
    xy = np.vstack([x_array, y_array])
    if estimate_type == 'kde':
        if bw_method == 'auto':
            z = gaussian_kde(xy)(xy)
        else:
            z = gaussian_kde(xy, bw_method=bw_method)(xy)
    if estimate_type == 'hist':
        hist, xedges, yedges, _, _ = get_hist_2d(x_array, y_array, bins_x=bins_x, bins_y=bins_y, stat='density')
        ix = np.digitize(x_array, xedges, right=False) - 1
        iy = np.digitize(y_array, yedges, right=False) - 1
        ix = np.clip(ix, 0, hist.shape[0] - 1)
        iy = np.clip(iy, 0, hist.shape[1] - 1)
        z = hist[ix, iy]
    return plt_colorful_scatter(ax, x, y, c=z, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, label=label, label_cmap_float=label_cmap_float, scatter_kwargs=scatter_kwargs, cbar=cbar, cbar_postion=cbar_position, cbar_kwargs=cbar_kwargs)

def plt_smooth_scatter_line(ax, x, y, frac=0.2, scatter_label=None, line_label='LOWESS', scatter_color=BLUE, scatter_alpha=FAINT_ALPHA, line_color=RED, scatter_kwargs=None, line_kwargs=None):
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    _print_invalid_value_warning('plt_smooth_scatter_line', x=x, y=y)
    x_smooth, y_smooth = lowess_smooth(x, y, frac=frac)
    return (plt_scatter(ax, x, y, color=scatter_color, label=scatter_label, alpha=scatter_alpha, **scatter_kwargs), plt_line(ax, x_smooth, y_smooth, color=line_color, label=line_label, **line_kwargs))

def plt_cdf(ax, data, label=None, color=BLUE, vert=True, **kwargs):
    _print_invalid_value_warning('plt_cdf', data=data)
    cdf = get_cdf(data)
    return plt_line(ax, cdf.x, cdf.y, label=label, color=color, vert=vert, **kwargs)

def plt_compare_cdf_ks(ax, data1, data2, label1=None, label2=None, color1=BLUE, color2=RED, vert=True, text_x=TEXT_X, text_y=TEXT_Y, fontsize=None, round_digit_dict=None, round_format_dict=None, show_list=None, text_kwargs=None, cdf_kwargs=None):
    if cdf_kwargs is None:
        cdf_kwargs = {}
    if fontsize is None:
        fontsize = FONT_SIZE
    plt_cdf(ax, data1, label=label1, color=color1, vert=vert, **cdf_kwargs)
    plt_cdf(ax, data2, label=label2, color=color2, vert=vert, **cdf_kwargs)
    r = get_ks_result(data1, data2)
    text_dict = {'KS': r['ks'], 'P': r['p']}
    add_text_by_dict(ax, text_dict=text_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=fontsize, text_x=text_x, text_y=text_y, text_kwargs=text_kwargs)

def plt_polygon_heatmap(ax, xy_dict, value_dict, mask=None, mask_color=MASK_COLOR, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cmap=CMAP, edgecolor=BLACK, cbar=True, cbar_position=None, cbar_label=None, fill=True, adjust_lim=True, polygon_kwargs=None, cbar_kwargs=None):
    if vmin is None:
        vmin = np.nanmin(list(value_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(value_dict.values()))
    polygon_kwargs = update_dict({}, polygon_kwargs)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    if mask is None:
        use_mask = False
    elif isinstance(mask, dict):
        if any(mask.values()):
            use_mask = True
        else:
            use_mask = False
    norm = get_norm(norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)
    for region, xy in xy_dict.items():
        if use_mask:
            if mask[region]:
                plt_polygon(ax, xy, color=None, facecolor=mask_color, edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
            else:
                plt_polygon(ax, xy, color=None, facecolor=cmap(norm(value_dict[region])), edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
        else:
            plt_polygon(ax, xy, color=None, facecolor=cmap(norm(value_dict[region])), edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
    if cbar:
        return add_side_colorbar(ax, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, cbar_label=cbar_label, cbar_position=cbar_position, use_mask=use_mask, mask_color=mask_color, **cbar_kwargs)

def plt_group_bar_dict(ax, data, width=None, colors=CMAP, vert=True, **kwargs):
    x_labels = list(data.keys())
    subgroup_labels = list(next(iter(data.values())).keys())
    for group, subgroups in data.items():
        if set(subgroups.keys()) != set(subgroup_labels):
            raise ValueError(f"'{group}'")
    y_values = [[subgroups[sub] for sub in subgroup_labels] for groups, subgroups in data.items()]
    return plt_group_bar(ax=ax, x=x_labels, y=y_values, label_list=subgroup_labels, width=width, colors=colors, vert=vert, **kwargs)

def sns_marginal_heatmap(ax, data, x_side_ax=None, y_side_ax=None, outside=True, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, x_color=BLUE, y_color=BLUE, heatmap_kwargs=None, bar_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True, inset_mode='fig'):
    heatmap_kwargs = update_dict(get_default_param(sns_heatmap), heatmap_kwargs)
    heatmap_kwargs['cbar_position'] = update_dict(CBAR_POSITION, heatmap_kwargs['cbar_position'])
    bar_kwargs = update_dict(get_default_param(plt_bar), bar_kwargs)
    bar_kwargs = update_dict(bar_kwargs, {'width': 1})
    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()
    if x_side_ax is None and y_side_ax is None:
        if outside:
            x_side_ax = add_side_ax(ax, position=x_side_ax_position, relative_size=x_side_ax_size, pad=x_side_ax_pad, sharex=ax, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
            y_side_ax = add_side_ax(ax, position=y_side_ax_position, relative_size=y_side_ax_size, pad=y_side_ax_pad, sharey=ax, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
        else:
            x_side_ax, y_side_ax, ax = split_with_double_marginal_ax(ax=ax, x_side_ax_position=x_side_ax_position, y_side_ax_position=y_side_ax_position, x_side_ax_size=x_side_ax_size, y_side_ax_size=y_side_ax_size, x_side_ax_pad=x_side_ax_pad, y_side_ax_pad=y_side_ax_pad, inset_mode=inset_mode)
    if heatmap_kwargs['cbar']:
        ax_width, ax_height = get_ax_size(ax)
        x_side_ax_width, x_side_ax_height = get_ax_size(x_side_ax)
        y_side_ax_width, y_side_ax_height = get_ax_size(y_side_ax)
        if heatmap_kwargs['cbar_position']['position'] == x_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += x_side_ax_pad + x_side_ax_height / ax_height
        if heatmap_kwargs['cbar_position']['position'] == y_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += y_side_ax_pad + y_side_ax_width / ax_width
    sns_heatmap(ax, local_data, **heatmap_kwargs)
    plt_bar(x_side_ax, x=np.arange(len(local_data.columns)) + 0.5, y=col_sum_df(local_data), **update_dict(bar_kwargs, {'color': x_color}))
    plt_bar(y_side_ax, x=np.arange(len(local_data.index)) + 0.5, y=row_sum_df(local_data), **update_dict(bar_kwargs, {'color': y_color, 'vert': False}))
    if rm_tick:
        rm_ax_tick(x_side_ax)
        rm_ax_tick(y_side_ax)
    if rm_spine:
        rm_ax_spine(x_side_ax)
        rm_ax_spine(y_side_ax)
    if rm_axis:
        rm_ax_axis(x_side_ax)
        rm_ax_axis(y_side_ax)
    if x_side_ax_position == 'bottom':
        x_side_ax.invert_yaxis()
    if y_side_ax_position == 'left':
        y_side_ax.invert_xaxis()
    return (ax, x_side_ax, y_side_ax)

def add_text_by_dict(ax, text_dict, show_list=None, round_digit_dict=None, round_format_dict=None, unit_dict=None, unit_mode='()', color_dict=None, fontsize=None, text_x=TEXT_X, text_y=TEXT_Y, text_process=None, text_kwargs=None):
    if show_list is None:
        show_list = list(text_dict.keys())
    if round_digit_dict is None:
        round_digit_dict = {key: ROUND_DIGITS for key in show_list}
    if round_format_dict is None:
        round_format_dict = {key: 'general' for key in show_list}
    if unit_dict is None:
        unit_dict = {key: None for key in show_list}
    if color_dict is None:
        color_dict = {key: BLACK for key in show_list}
    if fontsize is None:
        fontsize = FONT_SIZE
    text_process = update_dict(TEXT_PROCESS, text_process)
    if text_kwargs is None:
        text_kwargs = {}
    y_offset = 0.0
    for key in show_list:
        if key in text_dict:
            local_key = format_text(key, text_process=text_process)
            value = text_dict[key]
            round_value = round_float(value, round_digit_dict[key], round_format_dict[key], latex=True)
            if unit_dict[key] is not None:
                if unit_mode == '()':
                    round_value = f'{round_value} ({unit_dict[key]})'
                elif unit_mode == '[]':
                    round_value = f'{round_value} [{unit_dict[key]}]'
                else:
                    round_value = f'{round_value} {unit_dict[key]}'
            add_text(ax, f'{local_key}: {round_value}', x=text_x, y=text_y - y_offset, fontsize=fontsize, color=color_dict[key], **text_kwargs)
            y_offset += point_to_ax_proportion(ax, fontsize, axis='y')

def get_suitable_s(ax, num):
    ax_width, ax_height = get_ax_size(ax)
    ax_width_point = inch_to_point(ax_width)
    ax_height_point = inch_to_point(ax_height)
    s = np.pi * (np.min([ax_width_point, ax_height_point]) / np.sqrt(num)) ** 2
    s = np.min([s, 70.0])
    return s

def inch_to_point(inch):
    return inch * 72

def point_to_inch(point):
    return point / 72

def inch_to_ax_proportion(ax, inch, axis='x'):
    ax_width, ax_height = get_ax_size(ax)
    if axis == 'x':
        return inch / ax_width
    elif axis == 'y':
        return inch / ax_height
    else:
        raise ValueError(f"Unsupported axis={axis!r}; expected 'x' or 'y'")

def point_to_ax_proportion(ax, point, axis='x'):
    return inch_to_ax_proportion(ax, point_to_inch(point), axis)

def get_ax_size(ax):
    fig = ax.get_figure()
    if hasattr(fig, 'get_size_inches'):
        fig_width, fig_height = fig.get_size_inches()
    else:
        fig_width = fig.bbox.width / fig.dpi
        fig_height = fig.bbox.height / fig.dpi
    bbox = ax.get_position()
    ax_width = fig_width * bbox.width
    ax_height = fig_height * bbox.height
    return (ax_width, ax_height)

@iterate_over_axs
def set_ax_face_color(ax, color):
    ax.set_facecolor(color)

@iterate_over_axs
def rm_ax_spine(ax, spines_to_remove=None):
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right', 'left', 'bottom']
    if isinstance(spines_to_remove, str):
        spines_to_remove = [spines_to_remove]
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

@iterate_over_axs
def rm_ax_tick(ax, axis=None, which='both'):
    if axis is None or axis == 'both':
        axis = ['x', 'y']
    for tick in axis:
        if tick == 'x':
            ax.xaxis.set_tick_params(width=0, which=which)
        elif tick == 'y':
            ax.yaxis.set_tick_params(width=0, which=which)

@iterate_over_axs
def rm_ax_ticklabel(ax, axis=None, which='both'):
    if axis is None or axis == 'both':
        axis = ['x', 'y']
    for tick in axis:
        if tick == 'x':
            ax.tick_params(axis='x', which=which, labelbottom=False)
        elif tick == 'y':
            ax.tick_params(axis='y', which=which, labelleft=False)

@iterate_over_axs
def set_ax_tick(ax, ticks, labels, axis, which='major'):
    valid_tick_types = ('major', 'minor')
    if which not in valid_tick_types:
        raise ValueError(f'which must be one of {valid_tick_types}, got {which!r}.')
    if which == 'major':
        minor = False
    elif which == 'minor':
        minor = True
    if axis == 'x':
        ax.set_xticks(ticks, labels=labels, minor=minor)
    elif axis == 'y':
        ax.set_yticks(ticks, labels=labels, minor=minor)
    elif axis == 'z':
        ax.set_zticks(ticks, labels=labels, minor=minor)

@iterate_over_axs
def reduce_ax_tick(ax, axis='both', max_num=5):
    if axis in ['x', 'both']:
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_num))
        ax.xaxis.set_minor_locator(plt.NullLocator())
    if axis in ['y', 'both']:
        ax.yaxis.set_major_locator(plt.MaxNLocator(max_num))
        ax.yaxis.set_minor_locator(plt.NullLocator())

def share_axis(axs, sharex=True, sharey=True):
    if not isinstance(sharex, str):
        sharex = 'all' if sharex else 'none'
    if not isinstance(sharey, str):
        sharey = 'all' if sharey else 'none'
    if isinstance(axs, np.ndarray):
        if axs.ndim == 2:
            nrows, ncols = axs.shape
            for i in range(nrows):
                for j in range(ncols):
                    shared_with = {'none': None, 'all': axs[0, 0], 'row': axs[i, 0], 'col': axs[0, j]}
                    if shared_with[sharex] is not None:
                        axs[i, j].sharex(shared_with[sharex])
                    if shared_with[sharey] is not None:
                        axs[i, j].sharey(shared_with[sharey])
            return axs
    if sharex in ['row', 'col'] or sharey in ['row', 'col']:
        print("Warning: sharex and sharey can only be 'all' or 'none' when axs is not a 2D numpy array.")
    iterable_axs = get_iterable_ax(axs)
    for i, ax in enumerate(iterable_axs):
        if i == 0:
            pass
        else:
            if sharex == 'all':
                ax.sharex(iterable_axs[0])
            if sharey == 'all':
                ax.sharey(iterable_axs[0])
    if not isinstance(axs, plt.Axes):
        return rebuild_ax(iterable_axs, axs)
    else:
        return iterable_axs[0]

def share_axis_to_target(axs, target_ax, sharex=True, sharey=True):
    if sharex == 'none':
        sharex = False
    if sharey == 'none':
        sharey = False
    iterable_axs = get_iterable_ax(axs)
    for ax in iterable_axs:
        if sharex:
            ax.sharex(target_ax)
        if sharey:
            ax.sharey(target_ax)
    if not isinstance(axs, plt.Axes):
        return rebuild_ax(iterable_axs, axs)
    else:
        return iterable_axs[0]

@iterate_over_axs
def rm_ax_axis(ax):
    ax.axis('off')

def get_xylim_by_aspect_ratio(xlim, ylim, aspect_ratio=1.0):
    xmin, xmax = xlim
    ymin, ymax = ylim
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    x_range = xmax - xmin
    y_range = ymax - ymin
    current_aspect = y_range / x_range
    target_aspect = aspect_ratio
    if current_aspect > target_aspect:
        new_x_range = y_range / target_aspect
        new_xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
        new_ylim = ylim
    else:
        new_y_range = x_range * target_aspect
        new_ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)
        new_xlim = xlim
    return (new_xlim, new_ylim)

def label_title_process(x_label, y_label, z_label, title, text_process=None):
    text_process = update_dict(TEXT_PROCESS, text_process)
    return (format_text(x_label, text_process), format_text(y_label, text_process), format_text(z_label, text_process), format_text(title, text_process))

def add_axes_title(axes, title, text_process=None, title_size=None, pad=50.0, **kwargs):
    text_process = update_dict(TEXT_PROCESS, text_process)
    if title_size is None:
        title_size = TITLE_SIZE
    title = format_text(title, text_process)
    merged_ax = merge_ax(axes, rm_mode=False)
    merged_ax.set_title(title, pad=pad, fontsize=title_size, **kwargs)
    set_ax_face_color(merged_ax, color=TRANSPARENT)
    rm_ax_axis(merged_ax)

def suitable_tick_size(num_ticks, plt_size, tick_size=None, proportion=TICK_PROPORTION):
    if tick_size is None:
        tick_size = TICK_SIZE
    if num_ticks == 0:
        return tick_size
    suitable_tick_size = inch_to_point(plt_size) / num_ticks * proportion
    return min(suitable_tick_size, tick_size)

def get_tick_rotation(ax, axis):
    try:
        if axis == 'x':
            return ax.xaxis.get_ticklabels()[0].get_rotation()
        elif axis == 'y':
            return ax.yaxis.get_ticklabels()[0].get_rotation()
    except IndexError:
        if axis == 'x':
            return XTICK_ROTATION
        elif axis == 'y':
            return YTICK_ROTATION

@iterate_over_axs
def rm_ax_legend(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

@iterate_over_axs
def set_ax_legend(ax, loc=LEGEND_LOC, fontsize=None, bbox_to_anchor=None, text_process=None, rm_exist_legend=True, ncols=1, facecolor='inherit', edgecolor='0.8', labelcolor=None, frame_linewidth=None, **kwargs):
    if fontsize is None:
        fontsize = LEGEND_SIZE
    if rm_exist_legend:
        rm_ax_legend(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = (dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())
    if labels:
        labels = [format_text(label, text_process) for label in labels]
        legend = ax.legend(handles, labels, loc=loc, fontsize=fontsize, bbox_to_anchor=bbox_to_anchor, ncols=ncols, facecolor=facecolor, edgecolor=edgecolor, labelcolor=labelcolor, **kwargs)
        if frame_linewidth is None:
            frame_linewidth = LEGEND_FRAME_LINEWIDTH
        legend.get_frame().set_linewidth(frame_linewidth)

@iterate_over_axs
def set_ax(ax, xlabel=None, ylabel=None, zlabel=None, xlabel_pad=None, ylabel_pad=None, zlabel_pad=None, title=None, title_pad=None, text_process=None, title_size=None, label_size=None, tick_size=None, xtick=None, ytick=None, ztick=None, xtick_label=None, ytick_label=None, ztick_label=None, xtick_size=None, ytick_size=None, ztick_size=None, xtick_rotation=None, ytick_rotation=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, legend=True, legend_size=None, xlim=None, ylim=None, zlim=None, xlog=False, ylog=False, zlog=False, elev=None, azim=None, legend_loc=LEGEND_LOC, bbox_to_anchor=None, rm_exist_legend=True, legend_kwargs=None, tight_layout=False, reset_scale=False):
    if not isinstance(tight_layout, (bool, dict)):
        raise TypeError('tight_layout must be a bool or dict')
    text_process = update_dict(TEXT_PROCESS, text_process)
    legend_kwargs = update_dict({}, legend_kwargs)
    if xlabel_pad is None:
        xlabel_pad = LABEL_PAD
    if ylabel_pad is None:
        ylabel_pad = LABEL_PAD
    if zlabel_pad is None:
        zlabel_pad = LABEL_PAD
    if title_pad is None:
        title_pad = TITLE_PAD
    if title_size is None:
        title_size = TITLE_SIZE
    if label_size is None:
        label_size = LABEL_SIZE
    if tick_size is None:
        tick_size = TICK_SIZE
    if legend_size is None:
        legend_size = LEGEND_SIZE
    if xtick_rotation is None:
        xtick_rotation = get_tick_rotation(ax, 'x')
    if ytick_rotation is None:
        ytick_rotation = get_tick_rotation(ax, 'y')
    is_3d = isinstance(ax, Axes3D)
    xlabel = ax.get_xlabel() if xlabel is None else xlabel
    ylabel = ax.get_ylabel() if ylabel is None else ylabel
    if title is None:
        get_title = getattr(ax, 'get_title', None)
        title = get_title() if callable(get_title) else getattr(ax, 'title', None)
    if is_3d:
        zlabel = ax.get_zlabel() if zlabel is None else zlabel
    local_xlabel, local_ylabel, local_zlabel, local_title = label_title_process(xlabel, ylabel, zlabel, title, text_process)
    ax.set_xlabel(local_xlabel, fontsize=label_size, labelpad=xlabel_pad)
    ax.set_ylabel(local_ylabel, fontsize=label_size, labelpad=ylabel_pad)
    if is_3d:
        ax.set_zlabel(local_zlabel, fontsize=label_size, labelpad=zlabel_pad)
    ax.set_title(local_title, fontsize=title_size, pad=title_pad)
    if xtick is not None:
        set_ax_tick(ax, ticks=xtick, labels=xtick_label, axis='x')
    if ytick is not None:
        set_ax_tick(ax, ticks=ytick, labels=ytick_label, axis='y')
    if is_3d and ztick is not None:
        set_ax_tick(ax, ticks=ztick, labels=ztick_label, axis='z')
    if xtick_size is None:
        xtick_size = tick_size
    if ytick_size is None:
        ytick_size = tick_size
    if not is_3d and adjust_tick_size:
        width, height = get_ax_size(ax)
        xtick_size = suitable_tick_size(len(ax.get_xticks()), width, xtick_size, tick_proportion)
        ytick_size = suitable_tick_size(len(ax.get_yticks()), height, ytick_size, tick_proportion)
    ax.tick_params(axis='x', labelsize=xtick_size, rotation=xtick_rotation)
    ax.tick_params(axis='x', which='minor', rotation=xtick_rotation)
    ax.tick_params(axis='y', labelsize=ytick_size, rotation=ytick_rotation)
    ax.tick_params(axis='y', which='minor', rotation=ytick_rotation)
    if is_3d:
        if ztick_size is None:
            ztick_size = tick_size
        ax.tick_params(axis='z', labelsize=ztick_size)
    if xlog:
        ax.set_xscale('log')
    elif reset_scale:
        ax.set_xscale('linear')
    if ylog:
        ax.set_yscale('log')
    elif reset_scale:
        ax.set_yscale('linear')
    if is_3d:
        if zlog:
            ax.set_zscale('log')
        elif reset_scale:
            ax.set_zscale('linear')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if is_3d and zlim is not None:
        ax.set_zlim(zlim)
    if legend:
        set_ax_legend(ax, loc=legend_loc, fontsize=legend_size, bbox_to_anchor=bbox_to_anchor, text_process=text_process, rm_exist_legend=rm_exist_legend, **legend_kwargs)
    if tight_layout is True:
        plt.tight_layout()
    elif isinstance(tight_layout, dict):
        plt.tight_layout(**tight_layout)
    if is_3d:
        set_ax_view_3d(ax, elev=elev, azim=azim)

def set_ax_label(ax, label=None):
    if isinstance(ax, dict):
        if label is None:
            for k, v in ax.items():
                v.set_label(k)
        elif isinstance(label, dict):
            for k in ax.keys():
                if k in label:
                    ax[k].set_label(label[k])
    elif isinstance(ax, np.ndarray):
        if ax.ndim == 2:
            rows, cols = ax.shape
            if label is None:
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].set_label(f'row_{i}_col_{j}')
            elif isinstance(label, np.ndarray) and label.shape == ax.shape:
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].set_label(label[i, j])
        elif ax.ndim == 1:
            length = ax.shape[0]
            if label is None:
                for i in range(length):
                    ax[i].set_label(f'index_{i}')
            elif isinstance(label, np.ndarray) and label.shape == ax.shape:
                for i in range(length):
                    ax[i].set_label(label[i])
    elif isinstance(ax, plt.Axes):
        ax.set_label(label)

def get_suitable_fig_size(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, margin=None, adjust_params=None, adjust_params_custom=None, which=None):
    valid_modes = ('adjust_params_custom', 'adjust_params', 'margin')
    if which is None:
        raise ValueError(f'which must be explicitly specified as one of {valid_modes}')
    if which not in valid_modes:
        raise ValueError(f'Unsupported which={which!r}; expected one of {valid_modes}')
    if which == 'adjust_params' and adjust_params is None:
        raise ValueError("adjust_params must be provided when which='adjust_params'")
    margin = update_dict(MARGIN, margin)
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    if adjust_params_custom and which == 'adjust_params_custom':
        fig_width = ax_width * (ncols + adjust_params_custom['wspace'] * (ncols - 1)) + ax_width * (1 - adjust_params_custom['right'] + adjust_params_custom['left'])
        fig_height = ax_height * (nrows + adjust_params_custom['hspace'] * (nrows - 1)) + ax_height * (1 - adjust_params_custom['top'] + adjust_params_custom['bottom'])
    elif adjust_params and which == 'adjust_params':
        fig_width = ax_width * (ncols + adjust_params['wspace'] * (ncols - 1)) / (adjust_params['right'] - adjust_params['left'])
        fig_height = ax_height * (nrows + adjust_params['hspace'] * (nrows - 1)) / (adjust_params['top'] - adjust_params['bottom'])
    elif margin and which == 'margin':
        fig_width = ax_width / (margin['right'] - margin['left']) * ncols
        fig_height = ax_height / (margin['top'] - margin['bottom']) * nrows
    return (fig_width, fig_height)

def get_fig(width=AX_WIDTH, height=AX_HEIGHT, dpi=FIG_DPI, **kwargs):
    return plt.figure(figsize=(width, height), dpi=dpi, **kwargs)

def get_ax(fig=None, nrows=1, ncols=1, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, margin=None, squeeze=True, label='ax', subplots_params=None):
    if fig is None:
        fig = plt.gcf()
    margin = update_dict(MARGIN, margin)
    subplots_params = update_dict({}, subplots_params)
    if not isinstance(sharex, str):
        sharex = 'all' if sharex else 'none'
    if not isinstance(sharey, str):
        sharey = 'all' if sharey else 'none'
    ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            shared_with = {'none': None, 'all': ax[0, 0], 'row': ax[i, 0], 'col': ax[0, j]}
            ax[i, j] = fig.add_subplot(nrows, ncols, i * ncols + j + 1, sharex=shared_with[sharex], sharey=shared_with[sharey], **subplots_params)
    if rm_repeat_tick_label_when_share:
        if sharex in ['col', 'all']:
            for a in ax.flat:
                a._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ['row', 'all']:
            for a in ax.flat:
                a._label_outer_yaxis(skip_non_rectangular_axes=True)
    set_relative_ax_position(ax, nrows, ncols, margin=margin)
    if squeeze:
        ax = squeeze_ax(ax)
    set_ax_label(ax)
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax

def get_fig_ax(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, subplots_params=None, squeeze=True, margin=None, label='ax'):
    subplots_params = update_dict({}, subplots_params)
    margin = update_dict(MARGIN, margin)
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, margin=margin, which='margin')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height
    fig = get_fig(width=fig_width, height=fig_height)
    ax = get_ax(fig=fig, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, rm_repeat_tick_label_when_share=rm_repeat_tick_label_when_share, margin=margin, squeeze=squeeze, label=label, subplots_params=subplots_params)
    return (fig, ax)

def gfa(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, subplots_params=None, squeeze=True, margin=None, label='ax'):
    return get_fig_ax(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, fig_width=fig_width, fig_height=fig_height, sharex=sharex, sharey=sharey, rm_repeat_tick_label_when_share=rm_repeat_tick_label_when_share, subplots_params=subplots_params, squeeze=squeeze, margin=margin, label=label)

def get_gs_original(fig=None, nrows=1, ncols=1, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    if fig is None:
        fig = plt.gcf()
    return GridSpec(nrows, ncols, figure=fig, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)

def get_gs_inside_ax(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    left, right, bottom, top = get_ax_position_custom(ax)
    fig = ax.figure
    return get_gs_original(fig=fig, nrows=nrows, ncols=ncols, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)

def get_all_ax_from_gs(gs, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, squeeze=True, label='gs', **kwargs):
    if not isinstance(sharex, str):
        sharex = 'all' if sharex else 'none'
    if not isinstance(sharey, str):
        sharey = 'all' if sharey else 'none'
    ax = np.empty((gs.nrows, gs.ncols), dtype=object)
    for i in range(gs.nrows):
        for j in range(gs.ncols):
            shared_with = {'none': None, 'all': ax[0, 0], 'row': ax[i, 0], 'col': ax[0, j]}
            ax[i, j] = get_ax_from_gs(gs, index=(i, j), sharex=shared_with[sharex], sharey=shared_with[sharey], **kwargs)
    if rm_repeat_tick_label_when_share:
        if sharex in ['col', 'all']:
            for a in ax.flat:
                a._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ['row', 'all']:
            for a in ax.flat:
                a._label_outer_yaxis(skip_non_rectangular_axes=True)
    if squeeze:
        ax = squeeze_ax(ax)
    set_ax_label(ax)
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax

def get_ax_from_gs(gs, index=None, label='gs', **kwargs):
    fig = gs.figure
    if index is None:
        try:
            ax = fig.add_subplot(gs[:], **kwargs)
        except:
            ax = fig.add_subplot(gs, **kwargs)
    else:
        if not isinstance(index, tuple):
            raise ValueError('Index must be a tuple.')
        ax = fig.add_subplot(gs[index], **kwargs)
    set_ax_label(ax, index)
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax

def save_fig(fig, filename, formats=None, dpi=SAVEFIG_DPI, close=True, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES, filename_process=None, pkl=SAVEFIG_PKL, ax=None, **kwargs):
    if formats is None:
        formats = [SAVEFIG_FORMAT]
    else:
        formats = list(formats)
    filename_process = update_dict(FILENAME_PROCESS, filename_process)
    filename = format_filename(filename, filename_process)
    mkdir(os.path.dirname(filename))
    if filename.endswith(('.png', '.pdf', '.eps')):
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)
    if pkl:
        save_pkl((fig, ax), filename)
    if close:
        plt.close(fig)
        fig = None

def sns_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, log_scale=False, stat='probability', vert=True, **kwargs):
    if vert:
        sns.histplot(data, bins=bins, label=label, color=color, ax=ax, log_scale=log_scale, stat=stat, **kwargs)
    else:
        local_data = pd.DataFrame({'temp': data})
        original_ylabel = ax.get_ylabel()
        sns.histplot(local_data, y='temp', bins=bins, label=label, color=color, ax=ax, log_scale=log_scale, stat=stat, **kwargs)
        ax.set_ylabel(original_ylabel)
