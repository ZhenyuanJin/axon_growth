import os
import re
import sys
import pickle
import joblib
import shutil
import time
import datetime
import warnings
import multiprocessing
import inspect
import types
from functools import wraps, partial, lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO, BytesIO
import copy
import hashlib
import logging
import contextlib
import abc
import glob
from collections import defaultdict, Counter
import lmdb
import traceback
import tqdm
import natsort
import numpy as np
import scipy.stats as st
from scipy.stats import gaussian_kde, zscore
import scipy.sparse as sps
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter, ScalarFormatter, LogFormatter, LogFormatterExponent, LogFormatterMathtext, LogFormatterSciNotation, PercentFormatter
from matplotlib.colors import BoundaryNorm, Normalize, ListedColormap, SymLogNorm, LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D
import adjust_text_custom # use seed to make the random shifts reproducible


CF_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
BLUE = (0.01, 0.45, 0.70)
RED = (1.00, 0.27, 0.01)
GREEN = (0.01, 0.65, 0.50)
YELLOW = (0.99, 0.70, 0.01)
BLACK = (0.01, 0.01, 0.01)
WHITE = (1.00, 1.00, 1.00)
GRAY = (0.50, 0.50, 0.50)
ORANGE = (1.00, 0.50, 0.01)
PURPLE = (0.50, 0.01, 0.50)
CRIMSON = (220 / 255, 20 / 255, 60 / 255)
TRANSPARENT = (0.00, 0.00, 0.00, 0.00)
RANA = (235 / 255, 235 / 255, 235 / 255)
RYO = (93 / 255, 116 / 255, 165 / 255)
MARGIN = {'left': 0.2, 'right': 0.85, 'bottom': 0.2, 'top': 0.85}     # 默认图形边距
COMPACT_MARGIN = {'left': 0.1, 'right': 0.95, 'bottom': 0.1, 'top': 0.95}     # 边框较窄的图形边距
ADJUST_PARAMS_CUSTOM = {'left': 0.2, 'right': 0.85, 'bottom': 0.2, 'top': 0.85, 'wspace': 0.35, 'hspace': 0.35}     # 自定义的调整参数
FONT_SIZE = 15     # 默认字体大小
TITLE_SIZE = FONT_SIZE*2     # 默认标题字体大小
SUP_TITLE_SIZE = FONT_SIZE*3     # 默认总标题字体大小
LABEL_SIZE = FONT_SIZE*2     # 默认标签字体大小
LABEL_PAD = LABEL_SIZE/3     # label间距(间距的单位是字体大小)
TITLE_PAD = TITLE_SIZE/3     # 标题间距(间距的单位是字体大小)
TICK_SIZE = FONT_SIZE*4/3     # 默认刻度标签字体大小
LEGEND_SIZE = FONT_SIZE*4/3     # 默认图例字体大小
LINE_WIDTH = 3     # 默认线宽
MARKER_SIZE = LINE_WIDTH*2     # 默认标记大小,注意散点图的s是面积,markersize是直径
LEGEND_FRAME_LINEWIDTH = LINE_WIDTH / 3     # 默认图例外框线宽
ERRORBAR_LINEWIDTH = LINE_WIDTH     # 默认误差线线宽
AX_WIDTH = 5   # 默认图形宽度(单个图形)
AX_HEIGHT = 5   # 默认图形高度(单个图形)
FIG_SIZE = (AX_WIDTH / ( MARGIN['right'] - MARGIN['left'] ), AX_HEIGHT / ( MARGIN['top'] - MARGIN['bottom'] ))     # 默认图形大小
FIG_DPI = 300     # 默认图形分辨率
SAVEFIG_DPI = 300     # 默认保存图形分辨率
SAVEFIG_VECTOR_FORMAT = 'pdf'   # 默认保存图形的矢量图格式
SAVEFIG_RASTER_FORMAT = 'png'   # 默认保存图形的位图格式
SAVEFIG_FORMAT = SAVEFIG_VECTOR_FORMAT     # 默认保存图形的格式
TOP_SPINE = False     # 默认隐藏上方的轴脊柱
RIGHT_SPINE = False     # 默认隐藏右方的轴脊柱
LEFT_SPINE = True     # 默认显示左方的轴脊柱
BOTTOM_SPINE = True     # 默认显示下方的轴脊柱
LEGEND_LOC = 'upper right'      # 图例位置
MATHTEXT_FONTSET = 'stix'     # 使用数学文本的字体设置
PDF_FONTTYPE = 42     # 使pdf文件中的文字可编辑
PS_FONTTYPE = 42     # 适用于PostScript导出
PDF_USE14COREFONTS = False     # 禁用 14 种核心字体(强制嵌入自定义字体)
PAD_INCHES = 0.2     # 默认图形边距
USE_MATHTEXT = True     # 使用数学文本
USE_OFFSET = False     # 不使用偏移
AXES_LINEWIDTH = LINE_WIDTH     # 设置坐标轴线宽
TICK_MAJOR_WIDTH = 2    # x轴主刻度线的宽度
TICK_MINOR_WIDTH = 1    # x轴次刻度线的宽度
TICK_MAJOR_SIZE = 8     # x轴主刻度线的长度
TICK_MINOR_SIZE = 4     # x轴次刻度线的长度
TICK_DIRECTION = 'out'     # 刻度线的方向
RM_REPEAT_TICK_LABEL_WHEN_SHARE = True     # 当共享坐标轴时,去除重复的刻度标签
AX_FACECOLOR = (1., 1., 1., 0.9)     # 背景具有一定的透明度,防止遮挡
plt.rcParams['figure.figsize'] = FIG_SIZE   # 图形的大小
plt.rcParams['font.size'] = FONT_SIZE       # 字体大小
plt.rcParams['axes.labelsize'] = LABEL_SIZE       # 坐标轴标签的字体大小
plt.rcParams['axes.titlesize'] = TITLE_SIZE       # 标题的字体大小
plt.rcParams['axes.labelpad'] = LABEL_PAD       # 坐标轴标签与轴的距离
plt.rcParams['axes.titlepad'] = TITLE_PAD       # 标题与轴的距离
plt.rcParams['xtick.labelsize'] = TICK_SIZE       # x轴刻度标签的字体大小
plt.rcParams['ytick.labelsize'] = TICK_SIZE       # y轴刻度标签的字体大小
plt.rcParams['legend.fontsize'] = LEGEND_SIZE       # 图例的字体大小
plt.rcParams['axes.spines.top'] = TOP_SPINE    # 设定上方的轴脊柱
plt.rcParams['axes.spines.right'] = RIGHT_SPINE    # 设定右方的轴脊柱
plt.rcParams['axes.spines.left'] = LEFT_SPINE    # 设定左方的轴脊柱
plt.rcParams['axes.spines.bottom'] = BOTTOM_SPINE    # 设定下方的轴脊柱
plt.rcParams['axes.linewidth'] = AXES_LINEWIDTH    # 设置坐标轴线宽
plt.rcParams['lines.linewidth'] = LINE_WIDTH    # 线宽
plt.rcParams['lines.markersize'] = MARKER_SIZE    # 标记大小
plt.rcParams['patch.linewidth'] = LEGEND_FRAME_LINEWIDTH    # patch线宽(含legend外框)
plt.rcParams['savefig.format'] = SAVEFIG_FORMAT    # 保存图形的格式
plt.rcParams['figure.dpi'] = FIG_DPI      # 图形的分辨率
plt.rcParams['savefig.dpi'] = SAVEFIG_DPI      # 保存图像的分辨率
plt.rcParams['legend.loc'] = LEGEND_LOC      # 图例位置
plt.rcParams['mathtext.fontset'] = MATHTEXT_FONTSET     # 使用数学文本的字体设置
plt.rcParams['pdf.fonttype'] = PDF_FONTTYPE     # 使pdf文件中的文字可编辑
plt.rcParams['ps.fonttype'] = PS_FONTTYPE     # 适用于PostScript导出
plt.rcParams['pdf.use14corefonts'] = PDF_USE14COREFONTS     # 禁用14种核心字体(强制嵌入自定义字体)
plt.rcParams['savefig.pad_inches'] = PAD_INCHES     # 图形边距
plt.rcParams['axes.formatter.use_mathtext'] = USE_MATHTEXT     # 使用数学文本
plt.rcParams['axes.formatter.useoffset'] = USE_OFFSET     # 使用偏移
plt.rcParams['axes.linewidth'] = LINE_WIDTH     # 设置坐标轴线宽
plt.rcParams['xtick.major.width'] = TICK_MAJOR_WIDTH   # x轴主刻度线的宽度
plt.rcParams['xtick.minor.width'] = TICK_MINOR_WIDTH   # x轴次刻度线的宽度
plt.rcParams['ytick.major.width'] = TICK_MAJOR_WIDTH   # y轴主刻度线的宽度
plt.rcParams['ytick.minor.width'] = TICK_MINOR_WIDTH   # y轴次刻度线的宽度
plt.rcParams['xtick.major.size'] = TICK_MAJOR_SIZE  # x轴主刻度线的长度
plt.rcParams['xtick.minor.size'] = TICK_MINOR_SIZE   # x轴次刻度线的长度
plt.rcParams['ytick.major.size'] = TICK_MAJOR_SIZE  # y轴主刻度线的长度
plt.rcParams['ytick.minor.size'] = TICK_MINOR_SIZE   # y轴次刻度线的长度
plt.rcParams['xtick.direction'] = TICK_DIRECTION   # x轴刻度线的方向
plt.rcParams['ytick.direction'] = TICK_DIRECTION   # y轴刻度线的方向
plt.rcParams['axes.facecolor'] = AX_FACECOLOR  # 背景颜色
plt.rcParams['figure.subplot.wspace'] = ADJUST_PARAMS_CUSTOM['wspace']  # 调整子图之间的间距
plt.rcParams['figure.subplot.hspace'] = ADJUST_PARAMS_CUSTOM['hspace']  # 调整子图之间的间距
fm.fontManager.addfont(os.path.join(CF_DIR, 'font/arial.ttf'))  # Specify the path to your local Arial TTF file from https://github.com/gasharper/linux-fonts/blob/master/arial.ttf
plt.rcParams['font.family'] = 'Arial'  # Set Arial as the default font
SEED = 421
ROUND_DIGITS = 3
ROUND_FORMAT = 'general'
INF_POLICY = 'to_nan'
NAN_POLICY = 'drop'
TEXT_PROCESS = {'capitalize': True, 'replace_underscore': ' ', 'ignore_equation_underscore': True}
REPLACE_DOT = 'd'
FILENAME_PROCESS = {'replace_blank': '_', 'replace_dot': REPLACE_DOT} # replace_blank: 替换空格, replace_dot: 替换点
BAR_WIDTH = 0.8
BIN_NUM = 20
XTICK_ROTATION = 90
YTICK_ROTATION = 0
AUXILIARY_LINE_STYLE = '--'
PLT_CAP_SIZE_RATIO = 1.
PLT_CAP_SIZE = LINE_WIDTH * PLT_CAP_SIZE_RATIO
ARROW_STYLE = '-|>'
ARROW_HEAD_WIDTH = 0.3
ARROW_HEAD_LENGTH = 0.5
ARROW_PROPS = {'fc': RED, 'ec': RED, 'linewidth': LINE_WIDTH,
                      'arrowstyle': ARROW_STYLE, 'head_length': ARROW_HEAD_LENGTH, 'head_width': ARROW_HEAD_WIDTH}
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
TICK_PROPORTION = 0.9    # 刻度标签的拥挤程度,1代表完全贴住,见suitable_tick_size与adjust_ax_tick
SAVEFIG_PKL = False
CMAP = plt.cm.viridis
HEATMAP_CMAP = plt.cm.jet
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list("density_cmap", [RANA, BLUE])
TAG_CASE = 'lower'     # 默认tag大小写
TAG_PARENTHESES = False     # 默认tag是否加括号
AX_TAG_POS = (-0.2, 1.2)     # 默认坐标轴tag位置
TAG_SIZE = SUP_TITLE_SIZE     # 默认tag字体大小
TAG_VA = TEXT_VA     # 默认tag垂直对齐方式
TAG_HA = TEXT_HA     # 默认tag水平对齐方式
PRINT_WIDTH = 80     # 默认打印宽度
PRINT_CHAR = '-'     # 默认打印字符

def get_local_rng(seed=SEED):
    '''
    返回局部随机数生成器(rng:Random Number Generator)
    '''
    return np.random.RandomState(seed)

def get_time_seed():
    '''
    返回时间种子
    '''
    # 获取时间戳和微秒数
    timestamp = int(time.time())
    microsecond = datetime.datetime.now().microsecond

    # 结合时间戳和微秒数作为种子,为了防止超出随机数生成器的范围,取模
    return (timestamp * 1000000 + microsecond) % (2**31-1)

def get_time(char='_', unit='us'):
    """
    获取当前时间,支持秒/毫秒/微秒级精度

    :param char: 时间各部分分隔符,默认为下划线
    :param unit: 时间单位('s'-秒, 'ms'-毫秒, 'us'-微秒),默认为'us'
    :return: 格式化后的时间字符串
    """
    unit = unit.lower()
    valid_units = {'s', 'ms', 'us'}
    if unit not in valid_units:
        raise ValueError(f"Invalid unit: {unit}. Must be one of {valid_units}")

    # 单位到精度的映射
    unit_config = {
        's':  ('', 0),
        'ms': (f'{char}' + '%%03d', 3),  # 添加毫秒部分格式
        'us': (f'{char}' + '%%06d', 6)   # 添加微秒部分格式
    }

    now = datetime.datetime.now()

    # 基础时间格式
    base_format = f'%Y{char}%m{char}%d{char}%H{char}%M{char}%S'

    # 拼接完整格式
    suffix_format, precision = unit_config[unit]
    full_format = base_format + suffix_format

    # 处理不同精度需求
    if precision == 0:
        return now.strftime(full_format)
    else:
        truncated = now.microsecond // (10 ** (6 - precision))
        return now.strftime(full_format) % truncated

def sort_time_str(time_str_list, char='_', descending=False):
    def convert_to_datetime(time_str):
        parts = time_str.split(char)
        if len(parts[0]) == 4:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            second = int(parts[5])
            microsecond = 0
            if len(parts) == 7:
                fractional = parts[6]
                if len(fractional) == 3:
                    microsecond = int(fractional) * 1000
                elif len(fractional) == 6:
                    microsecond = int(fractional)
            return datetime.datetime(year, month, day, hour, minute, second, microsecond)
        else:
            return datetime.datetime.min

    return sorted(time_str_list, key=convert_to_datetime, reverse=descending)

def get_start_time():
    '''获取开始时间'''
    return time.perf_counter()

def _readable_time(seconds):
    if seconds < 60:
        return f"{seconds:.9f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} h"
    else:
        return f"{seconds / 86400:.2f} d"

def get_interval_time(start, title='', print_info=True, **kwargs):
    '''获取时间间隔'''
    interval = time.perf_counter() - start
    if print_info:
        if 'digits' in kwargs:
            print('do not use digits, this will be deprecated soon')
        if 'format_type' in kwargs:
            print('do not use format_type, this will be deprecated soon')
        # print_title(f'{title} takes {round_float(interval, digits, format_type)} s')
        print(f'{title} takes {_readable_time(interval)}')
    return interval

def func_timer(func):
    '''
    计时器修饰器
    使用例子:
    @func_timer
    def some_function():
        # some code here
    '''
    @wraps(func)  # 使用wraps装饰内部函数
    def wrapper(*args, **kwargs):
        start = get_start_time()
        result = func(*args, **kwargs)
        interval = get_interval_time(start, title=func.__name__, print_info=True)
        return result
    return wrapper

class Timer:
    '''
    计时器类
    '''
    def __init__(self, title='', print_info=True):
        self.title = title
        self.print_info = print_info
        self.start_time = None

    def start(self):
        self.start_time = get_start_time()
        if self.print_info:
            print_title(f'{self.title} starts')

    def end(self):
        if self.start_time is None:
            print('Please start the timer first')
            return None
        self.interval = get_interval_time(self.start_time, title=self.title, print_info=self.print_info)
        return self.interval

def flex_print_title(print_func=print, title=None, n=PRINT_WIDTH, char=PRINT_CHAR):
    '''灵活打印标题'''
    if title is not None:
        title = f'{title}'
        length_start = (n - len(title)) // 2
        length_end = n - len(title) - length_start
        print_func(length_start * char + title + length_end * char)
    else:
        print_func(char * n)

def print_title(title, n=PRINT_WIDTH, char=PRINT_CHAR):
    '''打印标题'''
    flex_print_title(print_func=print, title=title, n=n, char=char)

class Logger:
    '''
    需要利用logger.prt等来同时打印和记录log
    add方法只记录log不打印

    使用方法:

    - 使用logger的prt
    logger = Logger()
    logger.prt('message')

    - 利用python logging库实现,实时保存到文件
    logger.get_py_logger(filename='log.txt', name='logger')
    logger.py_logger即为python logging库的logger,可以按照python logging库的方式使用

    - 捕捉其他函数的print
    new_func = logger.capture(func)
    new_func(*args, **kwargs)

    - 捕捉其他class的print
    new_class = logger.capture(class)
    nc = new_class(*args, **kwargs)
    nc.some_method()

    - 捕捉某个代码块的print
    with Capturing(loggers=logger):
        some_code

    - 使用多个logger同时捕捉某个代码块的print
    with Capturing(loggers=[logger1, logger2]):
        some_code

    - 不使用with语句,直接捕捉print(记得手动调用end)
    cm = CaptureManager(loggers=[logger1, logger2]) 不需要调用enter,init中已经调用了
    some_code
    cm.end()

    - CaptureManager也可以使用with语句

    - 只添加log不打印
    logger.add('message')
    '''
    def __init__(self, n=PRINT_WIDTH, char=PRINT_CHAR, prt_mode=True, lite=True, print_title=True):
        '''
        prt_mode -- 是否打印
        '''
        # 自定义的log和参数
        self.log = []
        self.n = n
        self.char = char
        self.prt_mode = prt_mode
        self.lite = lite
        self.print_title = print_title
        # python logging库的logger(先设置为None)
        self.py_logger = None

    def get_py_logger(self, basedir, filename=None, name='', level=logging.DEBUG, format=None, datefmt='%Y-%m-%d %H:%M:%S', filemode='w', print_to_console=False, create_dir=True):
        '''
        注意: 特别不建议打开print_to_console,因为使用普通的抓取时,由于记录到logger中,这里会重复打印一遍
        '''
        self.basedir = basedir
        if create_dir:
            # 创建文件夹
            mkdir(self.basedir)

        # 获取filename
        if filename is None:
            filename = 'full.log'
        filename = safe_path_join(self.basedir, filename)

        # 设置format
        if format is None:
            if name == '':
                format = '%(asctime)s - %(levelname)s - %(message)s'
            else:
                format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # 获取python logging库的logger,注意,获得py_logger后,本类的所有方法才会记录到py_logger中
        self.py_logger = logging.getLogger(name)
        self.py_logger.propagate = False

        # 配置日志格式
        formatter = logging.Formatter(format, datefmt=datefmt)

        # 移除之前的所有处理器
        self.py_logger.handlers.clear()

        # 配置文件处理器
        if filename:
            file_handler = logging.FileHandler(filename, mode=filemode, delay=True)
            file_handler.setFormatter(formatter)
            self.py_logger.addHandler(file_handler)

        # 配置控制台处理器(如果设置了会同时打印到控制台)
        if print_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.py_logger.addHandler(console_handler)

        # 设置日志级别
        self.py_logger.setLevel(level)

    def add(self, *message, end='\n'):
        for i, m in enumerate(message):
            self.log.append(m)
            if i < len(message) - 1:
                s = ' '
            else:
                s = end
            if s is not None:
                self.log.append(s)
        if self.py_logger is not None:
            # 如果message只是一个'\n',则跳过
            if len(message) == 1 and message[0] == '\n':
                pass
            else:
                self.py_logger.info(' '.join(message))

def method_decorator(decorator):
    '''
    将装饰器应用于类的所有方法

    使用方法:
    已有一个decorator,可以使用如下方式将其应用到类的所有方法
    @method_decorator(decorator)
    '''
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
    """
    一个可以自定义输出消息的装饰器函数

    该装饰器可以用于装饰函数或类,在函数调用或类实例化时,
    会打印一个自定义的消息

    参数:
    message (str): 要输出的消息,可以使用 {0} 占位符来表示被装饰的对象名称

    返回:
    function: 返回一个装饰器函数

    使用示例:
    @message_decorator("The {0} needs improvement.")
    def my_function():
        pass

    @message_decorator("The {0} needs to be tested.")
    class MyClass:
        pass

    # 输出:
    # The my_function needs improvement.
    # The MyClass needs to be tested.
    """
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
            raise TypeError("This decorator supports only classes and functions.")
    return decorator

def to_be_improved(obj):
    return message_decorator("The '{0}' needs improvement.")(obj)

def func_tracker(func, print_func=print_title):
    '''
    当函数被调用时,打印函数名
    '''
    return message_decorator("Running: {0}", print_func=print_func)(func)

def init_tracker(func, print_func=print_title, cls_name=None):
    '''
    实例化时打印类名
    '''
    return message_decorator(f"Instance of {cls_name} created", print_func=print_func)(func)

def cls_tracker(cls):
    '''
    遍历类的所有方法,给每个方法添加装饰器,实例化时打印类名,当方法被调用时,打印方法名
    '''
    for name, method in cls.__dict__.items():
        if callable(method):
            if name == '__init__':
                setattr(cls, name, init_tracker(method, cls_name=cls.__name__))
            else:
                setattr(cls, name, func_tracker(method))
    return cls

def tracker_with_time(obj):
    '''
    tracker的增强版,在打印函数名的同时,还会打印时间
    '''
    if inspect.isfunction(obj):
        return func_timer(func_tracker(obj))
    elif inspect.isclass(obj):
        return method_decorator(func_timer)(cls_tracker(obj))

def get_detailed_error():
    exc_type, exc_value, exc_trace = sys.exc_info()
    if not exc_type:
        return

    # 获取错误基本信息
    error_name = exc_type.__name__
    error_msg = str(exc_value)

    # 获取最后一条堆栈记录
    stack_summary = traceback.extract_tb(exc_trace)
    last_frame = stack_summary[-1] if stack_summary else None

    # 构造详细日志
    error_info = f"""
    ======== 错误详情 ========
    错误类型: {error_name}
    错误信息: {error_msg}
    发生文件: {last_frame.filename if last_frame else 'unknown'}
    错误行号: {last_frame.lineno if last_frame else 0}
    问题代码: {last_frame.line if last_frame else 'unknown'}
    ======== 堆栈跟踪 ========
    {traceback.format_exc()}
    """
    return error_info

class FlexibleTry:
    '''
    示例:
    with FlexibleTry(enable_try=True):
        print(1/0)  # 会捕获异常并打印详细错误信息

    with FlexibleTry(enable_try=False):
        print(1/0)  # 会抛出异常

    with FlexibleTry(enable_try=True) as ft:
        print(1/0)  # 会捕获异常并打印详细错误信息
    print(ft.success)  # True or False, 是否成功
    print(ft.exception)  # 如果发生异常, 则为异常对象, 否则为None
    '''
    def __init__(self, enable_try=True, print_info=True):
        self.enable_try = enable_try
        self.print_info = print_info

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 没有发生异常
        if exc_type is None:
            self.success = True
            self.exception = None
            return False

        # 发生异常时的处理
        self.success = False
        self.exception = exc_val

        if self.enable_try and exc_type:
            # 当启用异常捕获且发生异常时,获取详细错误信息
            error_info = get_detailed_error()
            if self.print_info:
                print(error_info)  # 打印详细错误信息
            return True  # 吞掉异常

        return False  # 抛出异常

def caller_filename():
    '''
    打印调用此函数的文件名(带整个路径)

    如果是在Jupyter Notebook中运行,返回的是一个临时文件名,请使用current_file_ipynb函数获取当前文件名
    如果是在.py文件中运行,返回的是调用者的文件名(这也是为什么current_file需要显式复制这段代码,而不是调用这个函数,因为在common_functions.py中调用这个函数会返回common_functions.py)
    '''
    # 获取调用栈
    stack = inspect.stack()
    # 获取调用者的帧信息
    caller_frame = stack[1]
    # 获取调用者文件名
    caller_filename = caller_frame.filename
    # 打印文件名
    return caller_filename

def current_file_ipynb():
    '''获取当前文件名(带整个路径)'''
    from IPython import get_ipython
    ip = get_ipython()
    path = None
    if '__vsc_ipynb_file__' in ip.user_ns:
        path = ip.user_ns['__vsc_ipynb_file__']
    return path

def current_file():
    '''获取当前文件名(带整个路径)'''
    try:
        return current_file_ipynb()
    except:
        # 获取调用栈
        stack = inspect.stack()
        # 获取调用者的帧信息
        caller_frame = stack[1]
        # 获取调用者文件名
        caller_filename = caller_frame.filename
        # 打印文件名
        return caller_filename

def safe_path_join(*paths):
    '''
    os.path.join的再封装,保证不会因为中间的多余/或\导致路径不是预想的路径
    并且如果输入了空字符串或者None,则会自动忽略
    '''
    # 清理每个路径,去除前导和尾随的斜杠(只保留中间路径的斜杠)
    cleaned_paths = []
    for i, path in enumerate(paths):
        if path is None or path == "":
            continue
        else:
            if i == 0:
                # 第一个路径可以保留前导的斜杠(如果是绝对路径),但去掉尾部的斜杠
                cleaned_paths.append(path.rstrip("/\\"))
            else:
                # 后续路径要去掉前导和尾随斜杠
                cleaned_paths.append(path.strip("/\\"))

    # 使用 os.path.join 拼接清理后的路径
    return os.path.join(*cleaned_paths)

def pj(*args):
    '''
    极度缩写的safe_path_join
    '''
    return safe_path_join(*args)

def mkdir(fn):
    '''创建文件夹及中间文件夹'''
    os.makedirs(fn, exist_ok=True)

def delete_file(file_path):
    try:
        # 尝试删除文件
        os.remove(file_path)
        print(f"文件 '{file_path}' 已成功删除.")
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在.")
    except PermissionError:
        print(f"错误: 无权限删除文件 '{file_path}'.")
    except Exception as e:
        print(f"删除文件时发生意外错误: {e}")

def check_all_file_exist(*paths):
    '''
    检查所有文件路径是否存在
    参数: *paths (str): 可变数量的文件路径
    返回: bool: 仅当所有路径对应的文件/目录均存在时返回True,否则返回False
    '''
    return all(os.path.exists(path) for path in paths)

def find_files_with_any_extension(path):
    """
    查找指定路径的所有匹配文件(包括原始路径和任意后缀版本)

    参数:
        path (str): 文件路径

    返回:
        list: 包含所有匹配文件的完整路径列表(可能为空)
    """
    result = []

    # 如果原始路径存在(文件或目录),加入结果
    if os.path.exists(path):
        result.append(path)

    # 分离目录和文件名
    dir_path = os.path.dirname(path) or '.'  # 处理无目录的情况
    file_name = os.path.basename(path)

    # 构建匹配模式(文件名.*)
    pattern = os.path.join(dir_path, f"{file_name}.*")

    # 查找所有符合条件的普通文件(排除目录)
    for p in glob.glob(pattern):
        # 排除以点结尾的无效文件(如 "file.")
        if os.path.isfile(p) and not p.endswith('.'):
            # 避免重复添加原始路径(如果已存在)
            if p != path:
                result.append(p)

    return result

def check_all_file_exist_with_any_extension(*paths):
    """
    检查多个文件路径或该路径加任意后缀的文件是否存在

    参数:
        *paths (str): 可变数量的文件路径

    返回:
        bool: 所有路径都至少有一个匹配文件时返回True
    """
    for path in paths:
        # 如果找不到任何匹配文件,立即返回False
        if not find_files_with_any_extension(path):
            return False
    return True

def get_subfile(basedir, full=True):
    '''找到文件夹下的所有文件(非文件夹),full为True则返回全路径,否则只返回文件名'''
    if full:
        l = [os.path.join(basedir, f) for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]
    else:
        l = [f for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir, f))]
    l = natsort.natsorted(l)
    return l

def get_subdir(basedir, full=True):
    '''找到文件夹下的一级子文件夹,full为True则返回全路径,否则只返回文件夹名'''
    if full:
        l = [os.path.join(basedir, d) for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))]
    else:
        l = [d for d in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, d))]
    l = natsort.natsorted(l)
    return l

def compare_line_count(file1, file2):
    """
    比较两个文件的行数是否相同

    可以用于粗略检查两个文件是否相同,但有可能错判恰好有相同的行数但不相同的文件
    """
    with open(file1, 'r', encoding='utf-8') as f1:
        lines1 = sum(1 for _ in f1)

    with open(file2, 'r', encoding='utf-8') as f2:
        lines2 = sum(1 for _ in f2)

    return lines1 == lines2

def save_code_copy(destination_folder, code_file):
    '''
    将code保存为一个带时间戳的副本(code_file需要带后缀)

    示例:
    # 保存当前脚本
    cf.save_code_copy(basedir_code, cf.current_file())
    # 保存指定脚本
    cf.save_code_copy(basedir_code, r'../util/common_functions.py')
    '''
    os.makedirs(destination_folder, exist_ok=True)

    abs_code_path = os.path.abspath(code_file)

    code_name = os.path.basename(abs_code_path)
    file_prefix, file_extension = os.path.splitext(code_name)

    # 检查目标文件夹中是否存在行数相同的同名文件(前缀匹配)
    for existing_file in os.listdir(destination_folder):
        # 只检查前缀相同的文件
        if existing_file.startswith(file_prefix + '_') and existing_file.endswith(file_extension):
            existing_file_path = os.path.join(destination_folder, existing_file)
            # 如果找到行数相同的文件,直接返回不保存
            if compare_line_count(abs_code_path, existing_file_path):
                return

    # 生成带时间戳的文件名
    new_filename = f"{file_prefix}_{get_time()}{file_extension}"

    # 生成目标路径
    destination_path = os.path.join(destination_folder, new_filename)

    # 复制文件
    shutil.copy(abs_code_path, destination_path)

def save_dict_txt(filename, dict_data, key_to_save=None):
    """
    将字典数据保存为结构化文本文件

    参数：
    filename: 文件名(无需扩展名)
    dict_data: 要保存的字典数据
    key_to_save: 需要保存的键列表(默认保存全部键)
    """
    # 确定要保存的数据子集
    data_to_save = dict_data if key_to_save is None else {k: dict_data[k] for k in key_to_save}

    # 处理后缀
    if not filename.endswith('.txt'):
        filename += '.txt'

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 保存数据到文本文件
    with open(filename, 'w') as txt_file:
        def write_dict(d, indent):
            """递归写入字典的辅助函数"""
            for key, value in d.items():
                if isinstance(value, dict):
                    txt_file.write(' ' * indent + f'{key}:\n')
                    write_dict(value, indent + 4)  # 增加缩进
                else:
                    txt_file.write(' ' * indent + f'{key}: {value}\n')

        write_dict(data_to_save, 0)  # 从0缩进开始写入

def save_dict(dict_data, filename, format_list=None, key_to_save=None):
    '''保存字典到txt和pkl文件'''
    if format_list is None:
        format_list = ['txt', 'pkl']
    if key_to_save is None:
        key_to_save = list(dict_data.keys())

    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 假如filename有后缀,则添加到format_list中
    if filename.endswith('.txt'):
        format_list.append(filename.split('.')[-1])
        filename = os.path.splitext(filename)[0]
    if filename.endswith('.pkl') or filename.endswith('.pickle') or filename.endswith('.joblib'):
        format_list.append('.pkl')
        filename = os.path.splitext(filename)[0]

    # 保存到txt
    if 'txt' in format_list:
        save_dict_txt(filename, dict_data, key_to_save=key_to_save)

    # 保存到pkl
    if 'pkl' in format_list:
        save_pkl({k: dict_data[k] for k in key_to_save}, filename)

def get_load_function(format):
    '''
    根据格式获取合适的读取函数
    '''
    d = {'npy': load_array, 'npz': load_sps_array, 'pkl': load_pkl, 'joblib': load_pkl, 'pickle': load_pkl, '': load_dict_separate}
    return d[format]

def get_save_function_for_object(obj):
    '''
    根据数据类型获取合适的保存格式
    '''
    if isinstance(obj, np.ndarray):
        return save_array
    elif isinstance(obj, sps.spmatrix):
        return save_sps_array
    else:
        return save_pkl

def save_key_value_pair(key, value, save_func_dict, save_kwargs_dict, save_dir):
    '''
    用于save_dict_separate函数,保存键值对到文件
    '''
    str_key = hash_or_str(key)  # 将键转换为字符串或哈希值
    save_func = save_func_dict[key]  # 获取保存函数
    save_kwargs = save_kwargs_dict[key]  # 获取保存函数的参数
    save_func(value, os.path.join(save_dir, str_key), **save_kwargs)  # 保存值到文件
    return str_key, key

def save_dict_separate(dict_data, save_dir, save_func_dict=None, save_kwargs_dict=None, overwrite=False, save_as_txt=True, process_num=1, key_to_save=None, max_depth=0, current_depth=0, remove_after_save=False):
    """
    保存包含非字符串键的字典,将每个值保存为单独的文件,并保存键的映射关系

    支持嵌套使用(即字典的值为字典时,也会separate保存),但是不支持无限递归

    参数：
    dict_data (dict): 要保存的字典,字典的键可以是任何可哈希的类型
    save_dir (str): 保存文件的目标目录
    save_func_dict (dict): 指定每个键的保存函数,None则自动选择
    save_kwargs_dict (dict): 指定每个键的保存函数的参数,None则不输入参数
    overwrite (bool): 是否覆盖整个已存在的目标目录,默认为 False
    save_as_txt (bool): 是否保存为txt文件,默认为 True(txt保存的是整个字典,方便预览)
    process_num (int): 并行处理的进程数,默认为 1
    key_to_save (list): 指定要保存的键的列表,默认为 None,保存所有键
    max_depth (int, optional): 最大递归深度,默认为 0,表示不嵌套;不允许无限递归
    current_depth (int): 内部使用的当前递归深度,用户无需设置
    remove_after_save (bool): 是否删除已经保存的键值对
    """
    # 假设dict_data为空字典,则不保存
    if len(dict_data) == 0:
        print_title("Warning: The dictionary is empty. No files will be saved.", char='*')
        return

    if key_to_save is None:
        key_to_save = list(dict_data.keys())

    save_func_dict = fill_missing_key(save_func_dict, key_to_save, None)
    save_kwargs_dict = fill_missing_key(save_kwargs_dict, key_to_save, {})

    for k in key_to_save:
        if save_func_dict.get(k, None) is not None:
            continue  # 用户已指定保存函数,跳过
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

    # 保存键映射
    save_dict(metadata, os.path.join(save_dir, 'metadata'))

    # 保存原始字典的预览
    if save_as_txt:
        for k, v in metadata.items():
            if k in key_to_save:
                save_txt(dict_data[k], os.path.join(save_dir, 'preview', v))

    # 如果remove_after_save为True,则删除已经保存的键值对
    if remove_after_save:
        for k in key_to_save:
            del dict_data[k]

def save_dict_separate_merge_to_saved(dict_data, save_dir, save_func_dict=None, save_kwargs_dict=None, overwrite_separate=True, save_as_txt=True, process_num=1, key_to_save=None, max_depth=0, current_depth=0, remove_after_save=False):
    '''
    save_dict_separate,但是将保存的文件合并到已经存在的separate saved dict中,更新metadata

    注意:
    不允许save_dict_separate中的overwrite(会导致原先保存的dict被覆盖)
    可设置overwrite_separate,仅覆盖相同的键
    当overwrite_separate为False时,不允许原先dict和新dict中要key_to_save的键相同
    '''
    def _save():
        # 保存
        save_dict_separate(dict_data, save_dir, save_func_dict=save_func_dict, save_kwargs_dict=save_kwargs_dict, overwrite=False, save_as_txt=save_as_txt, process_num=process_num, key_to_save=key_to_save, max_depth=max_depth, current_depth=current_depth, remove_after_save=remove_after_save)

    # 如果metadata存在,则读取后更新metadata
    if check_all_file_exist_with_any_extension(os.path.join(save_dir, 'metadata')):
        # 读取已存在的metadata
        exist_metadata = load_pkl(os.path.join(save_dir, 'metadata'))

        # 检查是否存在相同的键
        if overwrite_separate is False:
            for k in key_to_save:
                if k in exist_metadata.values():
                    raise ValueError(f"Key '{k}' in key_to_save already exists in the existing metadata. This is not allowed when overwrite_separate is False.")

        # 保存(注意,这个和else是重复的,但是不能直接写到if else外面,否则只会保存新的metadata,旧的metadata会被覆盖)
        _save()

        # 更新metadata
        new_metadata = load_pkl(os.path.join(save_dir, 'metadata')) # 读取新保存的metadata
        metadata = update_dict(exist_metadata, new_metadata) # 合并metadata

        # 保存更新后的metadata
        save_dict(metadata, os.path.join(save_dir, 'metadata'))
    else:
        # 保存
        _save()

def part_load_dict_separate(load_dir, subfile, metadata, key_to_load):
    # 获取文件名和后缀
    filename, ext = os.path.splitext(subfile)
    # 跳过键映射文件
    if filename == 'metadata':
        return None, None
    if filename == 'preview':
        return None, None
    if filename in metadata.keys():
        if metadata[filename] in key_to_load:
            # 加载文件
            load_func = get_load_function(ext[1:])
            return metadata[filename], load_func(os.path.join(load_dir, subfile))
    return None, None

def _compatible_data_container(obj):
    '''
    定义一个修饰器,使读取函数兼容DataContainer
    '''
    @wraps(obj)
    def wrapper(*args, **kwargs):
        r = obj(*args, **kwargs)
        if isinstance(r, DataContainer):
            # 如果读取的对象是DataContainer,则不需要转换(当直接读pickle文件时会保持DataContainer)
            pass
        elif isinstance(r, dict):
            # 如果读取的对象是字典,则需要转换为DataContainer(当分开读取的时候会先读成字典)
            if '_config' in r:
                r = _dict_to_data_container(r)
        return r
    return wrapper

@_compatible_data_container
def load_dict_separate(load_dir, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    """
    从指定目录中加载字典,并还原原始的键和值类型.

    参数:
    load_dir (str): 包含文件及键映射的目录.
    key_to_load (list): 指定要加载的键的列表,默认为 None,加载所有键
    filter_str (str): 指定要加载的键的字符串,默认为 None,不过滤(如果希望过滤多个str,可以输入list或者tuple)
    filter_mode (str): 指定过滤模式,默认为 'include',可选 'include' 或 'exclude';'include'表示只加载包含指定字符串的键,'exclude'表示不加载包含指定字符串的键
    filter_logic (str): 指定过滤逻辑,默认为 'or',可选 'or' 或 'and';'or'表示只要有一个字符串匹配即可,'and'表示所有字符串都要匹配
    filter_func (function): 指定过滤函数,默认为 None,不过滤(过滤函数输出True则保留,False则删除)(当需要的逻辑无法通过字符串实现时,可以使用函数)
    process_num (int): 并行处理的进程数,默认为 1
    ensure_config (bool): 是否确保'_config'在key_to_load中,默认为 True,如果'_config'不在key_to_load中,则添加'_config'到key_to_load中

    返回:
    dict: 恢复的原始字典

    注意:
    filter_str 和 filter_func 可以同时使用,但进行的是一个串行的过滤操作,即先根据字符串过滤,再根据函数过滤
    """
    # 加载键的映射关系
    metadata = load_pkl(os.path.join(load_dir, 'metadata'))
    if key_to_load is None:
        # 注意metadata的key是哈希值或者字符串,value才是原始键
        key_to_load = list(metadata.values())

    # 根据过滤条件过滤键
    if filter_str is not None:
        # 确保 filter_str 是可迭代的
        if isinstance(filter_str, str):
            filter_str = [filter_str]

        if filter_mode == 'include':
            if filter_logic == 'or':
                key_to_load = [k for k in key_to_load if any(s in k for s in filter_str)]
            elif filter_logic == 'and':
                key_to_load = [k for k in key_to_load if all(s in k for s in filter_str)]
        elif filter_mode == 'exclude':
            if filter_logic == 'or':
                key_to_load = [k for k in key_to_load if not any(s in k for s in filter_str)]
            elif filter_logic == 'and':
                key_to_load = [k for k in key_to_load if not all(s in k for s in filter_str)]

    # 根据函数过滤键
    if filter_func is not None:
        key_to_load = [k for k in key_to_load if filter_func(k)]

    if ensure_config:
        # 确保'_config'在key_to_load中
        if '_config' in metadata.keys() and '_config' not in key_to_load:
            key_to_load.append('_config')

    loaded_data = {}

    r = multi_process_list_for(process_num=process_num, func=part_load_dict_separate, for_list=os.listdir(load_dir), kwargs={'load_dir':load_dir, 'metadata': metadata, 'key_to_load': key_to_load}, func_name=f'load dict from {load_dir}', for_idx_name='subfile', use_tqdm=False)
    # 遍历r并提取非None值
    while r:
        k, v = r.pop(0)  # 每次从列表前端弹出一个元素
        if k is not None:  # 检查键是否非None
            loaded_data[k] = v  # 存储到字典中

    # 检查loaded_data是否读取了所有key_to_load
    if set(loaded_data.keys()) != set(key_to_load):
        missing_keys = set(key_to_load) - set(loaded_data.keys())
        raise ValueError(f"Not all keys were loaded. Missing keys: {missing_keys}. Please check the files in {load_dir}.")
    return loaded_data

def load_dict_separate_merge_to_exist(exist_dict, load_dir, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    '''
    从指定目录中加载字典,并且与已存在的字典合并

    注意,exist_dict必须和loaded_dict拥有相同的结构,比如同时为dict或者同时为DataContainer

    在OrderedDataContainer中,'_config'里面有'param_map'
    由于这种嵌套性,_config不能直接像其他的key,value那样直接使用update

    例子:
    loaded_data = {'_config': {'param_map': {'a': '1'}}, ...}
    exist_dict['_config']['param_map'] = {'b': '2'}

    直接update
    exist_dict.update(config) # exist_dict['_config']['param_map'] = {'a': '1'}

    使用此函数, exist_dict['_config']['param_map'] = {'a': '1', 'b': '2'}
    '''
    loaded_data = load_dict_separate(load_dir, key_to_load=key_to_load, filter_str=filter_str, filter_mode=filter_mode, filter_logic=filter_logic, filter_func=filter_func, process_num=process_num, ensure_config=ensure_config)
    if '_config' in loaded_data.keys():
        config = loaded_data.pop('_config')
        exist_dict['_config']['param_map'].update(config['param_map'])
    exist_dict.update(loaded_data)
    return exist_dict

@_compatible_data_container
def load_dict_auto(filename, key_to_load=None, filter_str=None, filter_mode='include', filter_logic='or', filter_func=None, process_num=1, ensure_config=True):
    '''
    自动选择普通模式或者separate模式加载字典
    '''
    try:
        return load_dict_separate(filename, key_to_load=key_to_load, filter_str=filter_str, filter_mode=filter_mode, filter_logic=filter_logic, filter_func=filter_func, process_num=process_num, ensure_config=ensure_config)
    except:
        return load_pkl(os.path.join(filename))

@_compatible_data_container
def load_dict(*args, **kwargs):
    '''
    load_dict_auto的缩写
    '''
    return load_dict_auto(*args, **kwargs)

def pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir):
    '''
    弹出一部分参数,并且返回路径名
    '''
    local_dict_data = dict_data.copy()
    if value_dir_key is None:
        value_dir_key = []
    if both_dir_key is None:
        both_dir_key = []
    for key in value_dir_key:
        basedir = os.path.join(basedir, str(dict_data[key]))
        local_dict_data.pop(key)
    for key in both_dir_key:
        basedir = os.path.join(basedir, concat_str([key, str(dict_data[key])]))
        local_dict_data.pop(key)
    return local_dict_data, basedir

def save_timed_dir_dict(dict_data, basedir, dict_name, value_dir_key=None, both_dir_key=None, after_timedir='', current_time=None, format_list=None, pop=True):
    '''
    把一部分参数保存到路径名里,另一部分参数保存到文件里,并且在路径名里加入时间

    注意:
    pop为True,则保存在路径名里的参数会被弹出,不会保存到文件里;pop为False,则保存原先的字典
    '''
    if current_time is None:
        current_time = get_time()
    local_dict_data, basedir = pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir)
    timedir = os.path.join(basedir, current_time)
    dictdir = os.path.join(timedir, after_timedir)
    if pop:
        save_dict(local_dict_data, os.path.join(dictdir, dict_name), format_list)
    else:
        save_dict(dict_data, os.path.join(dictdir, dict_name), format_list)
    return timedir, dictdir

def save_pkl(obj, filename, format='joblib', compress=0, protocol=None, add_ext='auto'):
    '''
    保存对象到pkl文件(在本代码中pkl被认为是一种通用的保存格式,可以用joblib方法或者pickle方法保存,默认使用joblib方法[这可能会让人觉得有疑问,但是各类函数中的pkl都会优先使用joblib方法])
    虽然保存时优先使用joblib,但是加载时会兼容pkl,pickle,joblib三种格式,以对旧版本或者外源性的pkl文件进行兼容
    '''
    if format == 'joblib':
        save_joblib(obj, filename, compress=compress, protocol=protocol, add_ext=add_ext)
    elif format == 'pickle':
        save_pickle(obj, filename, add_ext=add_ext)

def load_pkl(filename):
    '''
    从pkl文件加载对象

    会采用各种方式,包括添加合适的后缀,直接读取;joblib和pickle都会尝试加载
    '''
    # 如果有后缀,且后缀已经在'.joblib','.pickle','.pkl'中,则直接加载
    if filename.endswith('.joblib'):
        return load_joblib(filename, add_ext=False)
    elif filename.endswith('.pickle'):
        return load_pickle(filename, add_ext=False)
    elif filename.endswith('.pkl'): # 兼容旧版本或者外源性的pkl文件
        return load_pickle(filename, add_ext=False)
    else: # 如果文件名没有后缀,或者后缀不在'.joblib','.pickle','.pkl'中,则尝试添加后缀并加载或者直接加载
        # 尝试添加后缀并加载
        for ext in ['.joblib', '.pickle', '.pkl']:
            if os.path.exists(filename + ext):
                return load_pkl(filename + ext)
        # 尝试直接加载(因为有可能输入的时候就有后缀)
        try:
            return load_joblib(filename, add_ext=False)
        except:
            try:
                return load_pickle(filename, add_ext=False)
            except:
                raise FileNotFoundError(f"File {filename} not found")

def save_pickle(obj, filename, add_ext='auto'):
    '''
    使用pickle保存对象

    参数:
    - add_ext:是否自动添加后缀,默认为 'auto',会自动添加后缀;如果不需要自动添加后缀,可以将此参数设置为 False
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 保存到pickle
    if add_ext == 'auto':
        if not filename.endswith('.pickle'):
            filename += '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename, add_ext='auto'):
    '''使用pickle加载对象,会自动添加后缀;由于后缀的多样性,当不需要自动添加后缀时,请将add_ext设置为False'''
    if add_ext == 'auto':
        ext_list = ['.pickle', '.pkl']
        detected_ext = False
        # 检查输入文件是否有指定后缀
        for ext in ext_list:
            if filename.endswith(ext):
                detected_ext = True
                break
        # 如果没有指定后缀,则尝试添加后缀并加载
        if not detected_ext:
            for ext in ext_list:
                if os.path.exists(filename + ext):
                    return load_pickle(filename + ext, add_ext=False)
            # 如果没有找到对应的文件,则抛出异常
            raise FileNotFoundError(f"File {filename} not found")
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_joblib(obj, filename, compress=0, protocol=None, add_ext='auto'):
    """
    使用joblib保存对象

    参数:
    - obj: 需要保存的对象
    - filename: 保存的文件名(如果没有以 .joblib 结尾,会自动添加)
    - compress: 压缩级别,默认为 0(范围为 0 到 9),0 表示不压缩,9 表示最大压缩
    - protocol: 序列化协议版本(默认使用 joblib 的默认协议).可以传递一个整数(如 3,4 或 5)
      表示 Python 的 Pickle 序列化协议版本
    - add_ext:是否自动添加后缀,默认为 'auto',会自动添加后缀;如果不需要自动添加后缀,可以将此参数设置为 False
    """
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    if add_ext == 'auto':
        # 如果文件名没有以 .joblib 结尾,自动添加后缀
        if not filename.endswith('.joblib'):
            filename += '.joblib'

    # 保存对象
    joblib.dump(obj, filename, compress=compress, protocol=protocol)

def load_joblib(filename, add_ext='auto'):
    """
    使用joblib加载对象,会自动添加后缀;由于后缀的多样性,当不需要自动添加后缀时,请将add_ext设置为False
    """
    # 检查文件是否以 .joblib 结尾
    if add_ext == 'auto':
        if not filename.endswith('.joblib'):
            filename += '.joblib'

    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")

    # 加载对象
    obj = joblib.load(filename)
    return obj

def search_dict_subdir(dict_data, basedir, pkl_name, ext='joblib', value_dir_key=None, both_dir_key=None, after_subdir='', ignore_key=None, pop=True):
    '''
    比较参数,如果参数不同,则返回False,如果参数相同,则返回对应文件夹,这个函数会遍历basedir下面的所有一级子文件夹,然后在每个一级子文件夹内的after_subdir文件夹下查找pkl文件

    适用场景:
    如果是pop_dict_get_dir保存,并且用时间作为文件夹,那么这个函数可以用来查找是否已经保存过这个dict,如果保存过,则返回对应的时间文件夹,和True,否则返回None和False

    返回:
    str: 时间文件夹的路径(没找到则为None)
    bool: 是否找到对应的文件
    '''
    all_timedir = []
    local_dict_data, basedir = pop_dict_get_dir(dict_data, value_dir_key, both_dir_key, basedir)
    if pop:
        pass
    else:
        local_dict_data = dict_data.copy()
    if not os.path.exists(basedir):
        return None, False
    for time_dir in get_subdir(basedir):
        pkl_dir = os.path.join(time_dir, after_subdir, f'{pkl_name}.{ext}')
        if os.path.exists(pkl_dir):
            exist_dict_data = load_pkl(pkl_dir)
            if compare_dict(local_dict_data, exist_dict_data, ignore_key):
                all_timedir.append(time_dir)
    only_time_str = []
    for timedir in all_timedir:
        only_time_str.append(os.path.basename(timedir))
    sorted_only_time_str = sort_time_str(only_time_str)
    sort_indices = get_sorted_index_from_sorted_list(only_time_str, sorted_only_time_str)
    all_timedir = sort_list_using_index(all_timedir, sort_indices)
    if len(all_timedir) > 0:
        if len(all_timedir) > 1:
            print(f"Warning: Multiple matching directories found for the given parameters. Returning the latest one: {all_timedir[-1]}")
        return all_timedir[-1], True
    else:
        return None, False

def compare_dict(dict1, dict2, ignore_key=None):
    '''
    比较两个字典,包括嵌套字典和NumPy数组,可选择忽略某些键
    '''
    # 处理忽略键
    if ignore_key is None:
        d1 = dict1.copy()
        d2 = dict2.copy()
    else:
        d1 = {k: v for k, v in dict1.items() if k not in ignore_key}
        d2 = {k: v for k, v in dict2.items() if k not in ignore_key}

    # 递归比较值
    def compare_values(v1, v2):
        if type(v1) is not type(v2):
            return False

        # 处理NumPy数组
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)

        # 处理嵌套字典
        elif isinstance(v1, dict) and isinstance(v2, dict):
            return compare_dict(v1, v2)

        # 处理列表
        elif isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(compare_values(item1, item2) for item1, item2 in zip(v1, v2))

        # 处理元组
        elif isinstance(v1, tuple) and isinstance(v2, tuple):
            if len(v1) != len(v2):
                return False
            return all(compare_values(item1, item2) for item1, item2 in zip(v1, v2))

        # 处理其他类型
        else:
            try:
                # 尝试直接比较
                return v1 == v2
            except (ValueError, TypeError):
                # 如果直接比较失败,转换为字符串比较
                print(f"Warning: Direct comparison of {v1} and {v2} failed, converting to string for comparison.")
                return str(v1) == str(v2)

    # 比较所有键值对
    if set(d1.keys()) != set(d2.keys()):
        return False
    for key in d1:
        if not compare_values(d1[key], d2[key]):
            return False
    return True

def load_df(filename, index_col=0, index_dtype=str, col_dtype=str):
    '''
    Load a DataFrame from a CSV, Excel file, or Pickle file, depending on the file's extension.

    Parameters:
    - filename: Name of the file to load the DataFrame from (with extension).
    - index_col: Column(s) to set as index. If None, defaults to pandas' behavior.
    - index_dtype: Data type for the index column, defaults to str.
    '''
    # Check if the file exists with the provided extension
    if os.path.exists(filename):
        if filename.endswith('.pkl') or filename.endswith('.joblib') or filename.endswith('.pickle'):
            df = pd.read_pickle(filename)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(
                filename, index_col=index_col, engine='openpyxl')
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename, index_col=index_col)

        df.index = df.index.astype(index_dtype)
        df.columns = df.columns.astype(col_dtype)

        # Convert all numerical columns to float
        df = df.apply(lambda x: x.astype(float)
                      if x.dtype.kind in 'iufc' else x)

        return df
    else:
        for ext in ['.xlsx', '.csv', '.pkl', '.joblib', '.pickle']:
            if os.path.exists(filename+ext):
                return load_df(filename+ext, index_col, index_dtype, col_dtype)
        raise ValueError(
            "File not found. Please check the file path and extension.")

def save_array(arr, filename):
    '''
    将数组保存到文件中.

    参数:
    arr: array
        要保存的数组对象.
    filename: str
        要保存的文件名,可以带有后缀或不带有后缀.

    返回:
    无返回值.
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.npy'):
        filename += '.npy'

    # 保存数组
    np.save(filename, arr)

def save_sps_array(arr, filename):
    '''
    将稀疏矩阵保存到文件中.

    参数:
    arr: sps.spmatrix
        要保存的稀疏矩阵对象.
    filename: str
        要保存的文件名,可以带有后缀或不带有后缀.

    返回:
    无返回值.
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.npz'):
        filename += '.npz'

    # 保存稀疏矩阵
    sps.save_npz(filename, arr)

def load_array(filename):
    '''
    从文件中加载数组.

    参数:
    filename: str
        要加载的文件名,可以带有后缀或不带有后缀.

    返回:
    array: array
        加载的数组对象.
    '''

    # 添加后缀
    if not filename.endswith('.npy'):
        filename += '.npy'

    # 加载数组
    return np.load(filename, allow_pickle=True)

def load_sps_array(filename):
    '''
    从文件中加载稀疏矩阵.

    参数:
    filename: str
        要加载的文件名,可以带有后缀或不带有后缀.

    返回:
    arr: sps.spmatrix
        加载的稀疏矩阵对象.
    '''
    # 添加后缀
    if not filename.endswith('.npz'):
        filename += '.npz'

    # 加载稀疏矩阵
    return sps.load_npz(filename)

def save_txt(data, filename):
    '''
    将数据保存到txt文件中
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.txt'):
        filename += '.txt'

    # 保存数据
    with open(filename, 'w') as f:
        f.write(str(data))

def save_list_txt(data, filename):
    '''
    将列表数据保存到txt文件中
    '''
    # 创建文件夹
    mkdir(os.path.dirname(filename))

    # 添加后缀
    if not filename.endswith('.txt'):
        filename += '.txt'

    # 保存数据
    with open(filename, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')

def load_txt(filename):
    '''
    从txt文件中加载数据

    参数:
    filename: str要加载的文件名,可以带有后缀或不带有后缀
    '''
    if not filename.endswith('.txt'):
        filename += '.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

def get_size_in_bytes(size, unit):
    '''
    统一转换为bytes
    '''
    units = {'bytes':1, 'kb':1024, 'mb':1024**2, 'gb':1024**3, 'tb':1024**4}
    return int(size * units[unit.lower()])

def init_lmdb(filename, size=1, unit='gb'):
    """初始化 LMDB 环境,自动根据现有数据库调整大小"""

    initial_map_size = get_size_in_bytes(size, unit)

    if os.path.exists(filename):
        try:
            # 尝试打开现有环境获取实际 map_size
            existing_env = lmdb.open(filename, readonly=True, lock=False)
            existing_map_size = existing_env.info()['map_size']
            existing_env.close()
            initial_map_size = max(initial_map_size, existing_map_size)
        except lmdb.Error:
            # 处理无法打开的情况(如文件损坏)
            pass

    return lmdb.open(filename,
                    map_size=initial_map_size,
                    subdir=True,
                    create=True,
                    readahead=True)

def _auto_expand(env, size, unit='gb', max_size=100, max_size_unit='gb'):
    """
    智能扩容机制,按几何倍增长
    """
    size_in_bytes = get_size_in_bytes(size, unit)
    max_size_in_bytes = get_size_in_bytes(max_size, max_size_unit)
    new_size_in_bytes = max(env.info()['map_size'], size_in_bytes) * 1.5
    new_size_in_unit = new_size_in_bytes / get_size_in_bytes(1, unit)
    if new_size_in_bytes > max_size_in_bytes:
        raise MemoryError(f'Need {new_size_in_unit:.2f}{unit} but max_size={max_size:.2f}{max_size_unit}, please set a larger size')
    env.set_mapsize(new_size_in_bytes)
    print(f"Auto-expanded to: {new_size_in_unit:.2f}{unit}")
    return new_size_in_unit

def _key_to_bytes(key):
    buffer = BytesIO()
    joblib.dump(key, buffer)
    return buffer.getvalue()

def _bytes_to_key(byte_key):
    buffer = BytesIO(byte_key)
    return joblib.load(buffer)

def save_dict_lmdb(data, filename, key_to_save=None, size=1, unit='gb', overwrite=True, max_size=100, max_size_unit='gb', remove_after_save=False, process_num=1):
    """根据 overwrite 决定是否覆盖"""
    if process_num != 1: # 为了与save_dict_separate兼容,这里支持输入process_num,但是不会生效
        print('LMDB has no multiprocess save')
    if key_to_save is None:
        key_to_save = list(data.keys())

    mkdir(os.path.dirname(filename))

    with init_lmdb(filename, size=size, unit=unit) as env:
        while size < max_size:
            try:
                with env.begin(write=True) as txn:
                    for key in key_to_save:
                        byte_key = _key_to_bytes(key)
                        buffer = BytesIO()
                        joblib.dump(data[key], buffer)
                        value = buffer.getvalue()
                        if overwrite or txn.get(byte_key) is None:
                            txn.put(byte_key, value)

                    # 保存所有key
                    all_keys = _get_all_keys_lmdb(txn)
                    save_list_txt(all_keys, os.path.join(filename, 'all_keys'))
                    save_joblib(all_keys, os.path.join(filename, 'all_keys'))

                    if remove_after_save:
                        for key in key_to_save:
                            del data[key]
                    return  # 成功保存后退出
            except lmdb.MapFullError:
                size = _auto_expand(env, size=size, unit=unit, max_size=max_size, max_size_unit=max_size_unit)
        raise lmdb.MapFullError(f"Auto expand to {size} {unit}, but exceed max_size={max_size} {max_size_unit}, please set a larger size")

@_compatible_data_container
def _load_dict_lmdb(filename, key_to_load):
    """
    从数据库加载数据

    为了性能上的考虑,只有在必要的时候才会得到all_keys,因为对于一个key相当多的lmdb,get_all_keys_lmdb_custom是比较耗费时间的
    """
    with init_lmdb(filename) as env:
        with env.begin() as txn:
            try:
                return {
                    key: joblib.load(BytesIO(txn.get(_key_to_bytes(key))))
                    for key in key_to_load
                }
            except Exception as e:
                all_keys = get_all_keys_lmdb_custom(filename)
                if is_sublist(key_to_load, all_keys):
                    raise ValueError(e)
                else:
                    raise KeyError(f"key_to_load {key_to_load} is not a sub list of all_keys in {os.path.join(filename, 'all_keys')}")

@_compatible_data_container
def load_dict_lmdb(filename, key_to_load=None, ensure_config=True, process_num=1):
    """
    从数据库加载数据

    为了性能上的考虑,只有在必要的时候才会得到all_keys,因为对于一个key相当多的lmdb,get_all_keys_lmdb_custom是比较耗费时间的
    """
    all_keys = None

    if key_to_load is None:
        all_keys = get_all_keys_lmdb_custom(filename)
        key_to_load = all_keys.copy()

    if ensure_config:
        if all_keys is None:
            all_keys = get_all_keys_lmdb_custom(filename)
        # 确保'_config'在key_to_load中
        if '_config' in all_keys and '_config' not in key_to_load:
            key_to_load.append('_config')

    divided_key_to_load = split_list(key_to_load, process_num)
    kwargs_list = []
    for k_to_load in divided_key_to_load:
        kwargs_list.append({'filename': filename, 'key_to_load': k_to_load})
    result_list = multi_process(process_num=process_num, func=_load_dict_lmdb, kwargs_list=kwargs_list, func_name='load_dict_lmdb')
    r = {}
    for result in result_list:
        r.update(result)
    return r

def load_dict_lmdb_merge_to_exist(exist_dict, filename, key_to_load=None, ensure_config=True, process_num=1):
    '''
    见load_dict_separate_merge_to_exist
    '''
    loaded_data = load_dict_lmdb(filename, key_to_load=key_to_load, ensure_config=ensure_config, process_num=process_num)
    if '_config' in loaded_data.keys():
        config = loaded_data.pop('_config')
        exist_dict['_config']['param_map'].update(config['param_map'])
    exist_dict.update(loaded_data)
    return exist_dict

def _get_all_keys_lmdb(txn):
    return [_bytes_to_key(byte_key) for byte_key, _ in txn.cursor()]

def get_all_keys_lmdb_custom(filename):
    '''
    从我保存的all_keys文件中获取所有键,极快
    '''
    all_keys = load_joblib(os.path.join(filename, 'all_keys'))
    return all_keys

def check_if_multiprocessing():
    current_process = multiprocessing.current_process()
    if current_process.name == 'MainProcess':
        return False
    else:
        return True

def split_list(lst, n):
    '''
    将列表尽量均等地分割为n个子列表.

    参数:
    - lst: list
        要分割的列表.
    - n: int
        子列表的数量.

    返回:
    list: 包含n个子列表的列表.
    '''
    # 计算每个子列表的长度
    length = len(lst)
    size = length // n
    remainder = length % n

    # 创建子列表
    divided_list = []
    start = 0
    for i in range(n):
        # 确定子列表的长度
        sublist_size = size + 1 if i < remainder else size
        # 添加子列表到结果列表中
        divided_list.append(lst[start:start + sublist_size])
        # 更新下一个子列表的起始位置
        start += sublist_size
    return divided_list

def _check_process_num_func_args_kwargs(process_num, func_list, args_list, kwargs_list):
    '''
    检查process_num, args_list, kwargs_list的长度是否符合要求
    注意:
    用户必须确保args_list,kwargs_list,func_list的长度相同
    process_num必须等于len(func_list)或者1
    '''
    len_set = {len(func_list), len(args_list), len(kwargs_list)}
    if len(len_set) > 1:
        raise ValueError("The length of func_list, args_list and kwargs_list must be the same.")
    if process_num != 1 and process_num != len(func_list):
        raise ValueError("The process_num must be equal to the length of func_list or 1.")

def basic_multi_process(process_num, func_list, args_list=None, kwargs_list=None, task_name=''):
    '''
    最基础的multi_process

    参数:
    - process_num: int, 并行处理的进程数
    - func_list: list, 要并行处理的函数列表
    - args_list: list, 函数的位置参数列表
    - kwargs_list: list, 函数的关键字参数列表
    - task_name: str, 任务的名称(也可以输入函数的名称等需要显示的信息)
    注意:
    用户必须确保args_list,kwargs_list,func_list的长度相同
    process_num必须等于len(func_list)或者1
    '''
    if args_list is None:
        args_list = [()] * len(func_list)
    if kwargs_list is None:
        kwargs_list = [{} for _ in func_list]

    _check_process_num_func_args_kwargs(process_num, func_list, args_list, kwargs_list)

    if process_num > 1 and check_if_multiprocessing():
        print_title('unable to use multiprocessing inside a multiprocessing process, use single process instead')
        process_num = 1

    if process_num != 1:
        print_title(f"Start {task_name} with {process_num} processes")
        results = []
        # 使用 ProcessPoolExecutor 进行多进程处理
        with ProcessPoolExecutor(max_workers=process_num) as executor:
            # 提交任务
            futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in zip(func_list, args_list, kwargs_list)]

            # 等待所有future对象按照提交的顺序完成,并收集结果
            for future in futures:
                with FlexibleTry() as ft:
                    results.append(future.result())
                if not ft.success:
                    results.append(None)
        print_title(f"Finish {task_name}")
        return results
    elif process_num == 1:
        return [func(*args, **kwargs) for func, args, kwargs in zip(func_list, args_list, kwargs_list)]

def _prepare_for_multi_process(process_num, func_list, args_list, kwargs_list):
    '''
    注意:
    当func_list,args_list,kwargs_list的len为1时,其会被自动广播到最大的一方
    假如args_list = [(1), (2)]这样的写法是不对的,至少要让里面成为元组,即args_list = [(1,), (2,)]
    假如已经在multi_process中,继续使用multi_process会自动转为单进程运行(此时args_list和kwargs_list会被flatten)
    '''
    # 确认输入的都是list或者None
    if (not isinstance(func_list, list)):
        raise ValueError("func_list must be a list")
    if (not isinstance(args_list, list)) and (args_list is not None):
        raise ValueError("args_list must be a list or None")
    if (not isinstance(kwargs_list, list)) and (kwargs_list is not None):
        raise ValueError("kwargs_list must be a list or None")

    # 处理None(args_list和kwargs_list本身为None)
    if args_list is None:
        args_list = [()]
    else:
        args_list = args_list.copy()
    if kwargs_list is None:
        kwargs_list = [{}]
    else:
        kwargs_list = kwargs_list.copy()

    # 处理长度不一致的情况
    max_length = max(len(func_list), len(args_list), len(kwargs_list))
    if len(func_list) != max_length:
        if len(func_list) == 1:
            func_list = func_list * max_length
        else:
            raise ValueError("The length of func_list must be equal to max_length or 1.")
    if len(args_list) != max_length:
        if len(args_list) == 1:
            args_list = args_list * max_length
        else:
            raise ValueError("The length of args_list must be equal to max_length or 1.")
    if len(kwargs_list) != max_length:
        if len(kwargs_list) == 1:
            kwargs_list = kwargs_list * max_length
        else:
            raise ValueError("The length of kwargs_list must be equal to max_length or 1.")

    # 处理None(args_list和kwargs_list内部的None)
    for i, args in enumerate(args_list):
        if args is None:
            args_list[i] = ()
    for i, kwargs in enumerate(kwargs_list):
        if kwargs is None:
            kwargs_list[i] = {}

    return process_num, func_list, args_list, kwargs_list

def _run_multi_func(func_list, args_list, kwargs_list):
    '''
    运行多函数
    '''
    results = []
    for func, args, kwargs in zip(func_list, args_list, kwargs_list):
        results.append(func(*args, **kwargs))
    return results

def multi_process_for_different_func(process_num, func_list, args_list=None, kwargs_list=None, task_name=''):
    '''
    多进程并行处理不同的函数,相较basic_multi_process,增添了以下功能

    自动广播:如果func_list,args_list,kwargs_list的len为1时,其会被自动广播到最大的一方(如果为None,则也自动设置)
    自动切分:process_num不再被限制为只能是1或者len(func_list),而是可以是任意值,但是会自动切分func_list,args_list,kwargs_list
    '''
    process_num, func_list, args_list, kwargs_list = _prepare_for_multi_process(process_num, func_list, args_list, kwargs_list)

    divided_func_list = split_list(func_list, process_num)
    divided_args_list = split_list(args_list, process_num)
    divided_kwargs_list = split_list(kwargs_list, process_num)

    new_func_list = [_run_multi_func] * process_num
    new_args_list = [()] * process_num
    new_kwargs_list = [{'func_list': divided_func_list[i], 'args_list': divided_args_list[i], 'kwargs_list': divided_kwargs_list[i]} for i in range(process_num)]
    return flatten_list(basic_multi_process(process_num, func_list=new_func_list, args_list=new_args_list, kwargs_list=new_kwargs_list, task_name=task_name), level=1)

def multi_process(process_num, func, args_list=None, kwargs_list=None, func_name=''):
    '''
    多进程并行处理函数,内部使用multi_process_for_different_func

    参数:
    - process_num: int,并行处理的进程数(由于multi_process状态下,代码错误的提示难以看出错误位置,在测试时可以先把process_num设置为1,这时会按照正常默认方式运行和报错)
    - func: function,要并行处理的函数
    - args_list: list,函数的位置参数列表
    - kwargs_list: list,函数的关键字参数列表
    - func_name: str,函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    假如args_list = [(1), (2)]这样的写法是不对的,至少要让里面成为元组,即args_list = [(1,), (2,)]
    假如已经在multi_process中,继续使用multi_process会自动转为单进程运行(此时args_list和kwargs_list会被flatten)

    假如输入的args_list和kwargs_list都不包含多个元素,最终会被广播到process_num的数量
    否则,process_num只代表分割的数量,而不是重复的数量

    multi_process下要尽量避免使用全局随机和输入同一个rng,如果函数一样,多个process会有一样的结果,推荐的使用的方式是输入不同的seed在内部产生rng
    '''
    def _get_input_l_len(l):
        if l is None:
            return 0
        elif isinstance(l, list):
            return len(l)
        else:
            raise ValueError("args_list and kwargs_list must be list or None")

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
    '''
    多进程并行处理for循环,for循环形式为for i in for_list

    参数:
    - process_num: int, 并行处理的进程数
    - func: function, 要并行处理的函数
    - args: 函数的位置参数(不推荐,因为idx在func中的位置不确定)
    - kwargs: 函数的关键字参数
    - func_name: str, 函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    只有当for循环每个之间独立时才能使用这个函数
    如果需要使用items()方法,请使用multi_process_items_for;如果需要使用enumerate()方法,请使用multi_process_enumerate_for;此处尚未支持zip()方法,但是zip也可以通过普通for循环实现
    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if for_list is None:
        raise ValueError("for_list must not be None")
    for_list = list(for_list)   # 防止for_list是生成器,比如range(10)

    if process_num > len(for_list):
        print_title(f'Reduce process_num from {process_num} to {len(for_list)} as process_num > len(for_list)')
        process_num = len(for_list)

    divided_list = split_list(for_list, process_num)
    args_list = [(func, divided, for_idx_name, False)+args for divided in divided_list]
    if use_tqdm:
        args_list[0] = (func, divided_list[0], for_idx_name, True) + args # 只有第一个进程显示tqdm
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
    '''
    多进程并行处理for循环,for循环形式为for key, value in for_dict.items()

    参数:
    - process_num: int, 并行处理的进程数
    - func: function, 要并行处理的函数(其会接收for_key_name和for_value_name作为关键字参数)
    - args: 函数的位置参数(不推荐,因为key,value在func中的位置不确定)
    - kwargs: 函数的关键字参数
    - func_name: str, 函数的名称(也可以输入任务的名称等需要显示的信息)

    注意:
    只有当for循环每个之间独立时才能使用这个函数

    示例:
    def func(k, v, x, y):
        print(k, v, x, y)

    for_dict = {'a': 1, 'b': 2, 'c': 3}
    multi_process_items_for(2, func, args=(), kwargs={'x': 3, 'y': 3}, for_dict=for_dict, for_key_name='k', for_value_name='v')
    '''
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    if process_num > len(for_dict):
        print_title(f'Reduce process_num from {process_num} to {len(for_dict)} as process_num > len(for_dict)')
        process_num = len(for_dict)

    divided_key_list = split_list(list(for_dict.keys()), process_num)
    divided_value_list = split_list(list(for_dict.values()), process_num)
    args_list = [(func, divided_key, divided_value, for_key_name, for_value_name, False)+args for divided_key, divided_value in zip(divided_key_list, divided_value_list)]
    if use_tqdm:
        args_list[0] = (func, divided_key_list[0], divided_value_list[0], for_key_name, for_value_name, True) + args # 只有第一个进程显示tqdm
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
        # 在每个标签前后添加括号
        tags = ['(' + tag + ')' for tag in tags]

    return tags

def get_decimal_num(number, abs_tol=1e-5, rel_tol=1e-5):
    """
    计算浮点数的有效小数位数.如果某一位的小数值相对于数本身小于tol,则从该位开始忽略.

    :param number: 输入的浮点数
    :param abs_tol: 容忍度,决定从哪一位开始忽略,默认值为 1e-5(如果不想检测abs_tol,可以设置为None)
    :param rel_tol: 相对容忍度,决定从哪一位开始忽略,默认值为 1e-5(如果不想检测rel_tol,可以设置为None)
    :return: 小数位数
    """
    # 记录初始的小数位数
    decimal_count = 0

    # 我们循环检查小数位,逐位扩大10倍,直到发现某一位可以忽略
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

        # 否则继续检查下一位
        decimal_count += 1

    return decimal_count

def get_max_decimal_num(number_list, **kwargs):
    decimal_count_list = [get_decimal_num(number, **kwargs) for number in number_list]
    return max(decimal_count_list)

def format_text(text, text_process=None):
    '''
    格式化文本(主要是为了画图美观).

    当 ignore_equation_underscore=True 时, 按 Matplotlib Mathtext 的$...$ 规则保留公式内的下划线. 如果未转义的 $ 数量为奇数, 整个字符串按普通文本处理.
    '''
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
                    if char == '_' and not in_math and replace_underscore:
                        processed_chars.append(replace_underscore)
                    else:
                        processed_chars.append(char)
                parts = [''.join(processed_chars)]
            else:
                parts = [text.replace('_', replace_underscore)
                         if replace_underscore else text]
        else:
            parts = [text.replace('_', replace_underscore)
                     if replace_underscore else text]

        # 处理大写转换
        if capitalize:
            processed_parts = [part[0].capitalize(
            ) + part[1:] if part else "" for part in parts]
        else:
            processed_parts = parts

        return "".join(processed_parts)

def sci_str_to_latex(sci_str):
    '''
    将科学计数法字符串转换为LaTeX字符串

    例子:
    sci_str_to_latex('1.23e-4')  # 返回 '$1.23 \\times 10^{-4}$'

    注意:
    此函数与sci_float_to_latex不同,不仅在于输入的不同(一个字符串一个浮点数),还在于内部的处理方式不同(此函数假定了浮点数已经被python原生的round scientific notation表示并以10为底,而sci_float_to_latex手动实现了这个过程,可以以其他数为底)
    '''
    match = re.match(r"([+-]?\d*\.\d+)e([+-]?\d+)", sci_str)
    if match:
        base, exponent = match.groups()
        # Format the LaTeX string
        latex_str = f"${base} \\times 10^{{{exponent}}}$"
        return latex_str
    else:
        raise ValueError('The input string is not in scientific notation.')

def round_float(number, digits=ROUND_DIGITS, format_type=ROUND_FORMAT, latex=False):
    '''
    Rounds a float to a given number of digits and returns it in specified format.

    Parameters:
    - number: float, the number to be rounded.
    - digits: int, the number of digits to round to.
    - format_type: str, the format type for the output ('standard', 'scientific', 'percent', 'general'). 'general' means choose standard or scientific based on the length of the string representation.
    - latex: bool, whether to return the rounded number as a LaTeX string for scientific notation.

    Returns:
    - str, the rounded number as a string in the specified format.
    '''
    if format_type == 'general':
        result = f"{number:.{digits}g}"
        if 'e' in result:
            return round_float(number, digits=digits, format_type='scientific', latex=latex)
        else:
            return result
    elif format_type == 'standard':
        return f"{number:.{digits}f}"
    elif format_type == 'scientific':
        if latex:
            return sci_str_to_latex(f"{number:.{digits}e}")
        else:
            return f"{number:.{digits}e}"
    elif format_type == 'percent':
        return f"{number:.{digits}%}"

def get_coefficient_exponent(number, base=10):
    '''
    将浮点数转换为系数和指数的形式
    '''
    exponent = int(np.floor(np.log(abs(number)) / np.log(base)))
    coefficient = number / (base ** exponent)
    return coefficient, exponent

def sci_float_to_latex(number, base=10, base_str='10', round_digits=0):
    '''
    假设是coefficient乘以base^exp的形式,返回科学计数法的latex字符串

    注意:
    默认状态下,假设了number的形式,并非通用,只适用于整数乘以base^exp的形式,一般来说,是用在ticks的标签上
    当然,也可以用在其他地方,只要符合coefficient乘以base^exp的形式(但是这时需要调整round_digits)
    '''
    if number == 0:
        return r"$0$"  # 0 doesn't have an exponential form

    coefficient, exp = get_coefficient_exponent(number, base)

    # 当在整数模式下使用,良好的处理整数1和-1
    if np.allclose(round_digits, 0):
        if np.allclose(coefficient, 1):
            return fr"${base_str}^{{{exp}}}$"
        elif np.allclose(coefficient, -1):
            return fr"$-{base_str}^{{{exp}}}$"
    # 当在小数模式下使用,不处理整数1和-1,直接返回
    return fr"${coefficient:.{round_digits}f} \times {base_str}^{{{exp}}}$"

def format_filename(filename, file_process=None):
    '''
    格式化文件名(目前是为了去除空格,格式化浮点数的小数点没有加入此函数)
    '''
    file_process = update_dict(FILENAME_PROCESS, file_process)
    if filename is None:
        return None
    else:
        return filename.replace(' ', file_process['replace_blank'])

def concat_str(strs, sep='_', rm_double_sep=True, ignore_none=True, ignore_empty=True, convert_non_str=True):
    '''
    连接字符串列表,并使用指定的分隔符连接

    Parameters:
    - strs: 要连接的字符串列表
    - sep: 用作分隔符的字符串,默认为'_'
    - rm_double_sep: 是否移除重复的分隔符,默认为True
    - ignore_none: 是否忽略None值,默认为True
    - ignore_empty: 是否忽略空字符串,默认为True
    - convert_non_str: 是否将非字符串类型转换为字符串,默认为True

    Returns:
    - 连接后的字符串
    '''
    # 如果 ignore_none 为 True,过滤掉 None 值
    if ignore_none:
        strs = [s for s in strs if s is not None]

    # 如果 ignore_empty 为 True,过滤掉空字符串
    if ignore_empty:
        strs = [s for s in strs if s != '']

    # 如果 convert_non_str 为 True,将非字符串类型转换为字符串
    if convert_non_str:
        strs = [str(s) for s in strs]

    # 连接字符串
    result = sep.join(strs)

    # 如果设置了 rm_double_sep 为 True,移除重复的分隔符
    if rm_double_sep:
        while sep*2 in result:
            result = result.replace(sep*2, sep)

    return result

def cat(*args, sep='_', rm_double_sep=True, ignore_none=True, ignore_empty=True, convert_non_str=True):
    '''
    连接字符串列表,并使用指定的分隔符连接(简化版)
    '''
    return concat_str(strs=args, sep=sep, rm_double_sep=rm_double_sep, ignore_none=ignore_none, ignore_empty=ignore_empty, convert_non_str=convert_non_str)

def hash_or_str(key, sep='_', replace_dot=REPLACE_DOT):
    """将键转换为字符串,如果复杂则生成哈希值."""
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
    '''
    将元组转换为字符串,并使用指定的分隔符连接

    replace_dot: 则将.替换为replace_dot以应对浮点数转换时出现的'.'

    有时候画图需要按照xlim,ylim等作为文件名,这个函数可以将xlim,ylim等转化为字符串
    '''
    result = sep.join(map(str, t))
    if replace_dot:
        result = result.replace('.', replace_dot)
    return result

def fill_missing_key(d, key_list, fill_value=None):
    '''
    已弃用: 为字典补充缺失键,后续版本将删除此函数.

    如果d为None,返回包含key_list中所有键的新字典;否则浅拷贝d,
    再为其中的缺失键填入fill_value.

    警告:
    每个缺失键会直接引用同一个fill_value.当fill_value是dict、list等
    可变对象时,修改一个键的值会影响其他缺失键.需要为多个键分别创建
    空字典时,请使用fill_dict_kwargs.
    '''
    warnings.warn(
        'fill_missing_key已弃用,并将在后续版本删除;'
        '需要为多个键创建独立空字典时请使用fill_dict_kwargs.',
        FutureWarning,
        stacklevel=2,
    )
    if d is None:
        return {k: fill_value for k in key_list}
    new_d = d.copy()
    for k in key_list:
        if k not in new_d:
            new_d[k] = fill_value
    return new_d

def create_dict(keys, values=None, default_value=None, deep_copy=True):
    """
    创建字典

    参数:
    - keys: list, 字典的键
    - values: list, 字典的值
    - default_value: 任意类型, 默认值
    - deep_copy: bool, 是否深拷贝,默认为True

    注意:
    - 如果values为None,则使用default_value作为默认值
    - 如果values不为None,则使用values作为值
    - 用这个函数创建字典时,默认值和值都是深拷贝的,不会产生同时变化的问题
    """
    if deep_copy:
        if values is None:
            return {key: copy.deepcopy(default_value) for key in keys}
        else:
            return {key: copy.deepcopy(value) for key, value in zip(keys, values)}
    else:
        if values is None:
            return {key: default_value for key in keys}
        else:
            return {key: value for key, value in zip(keys, values)}

def filter_dict(d, keys):
    '''过滤字典'''
    return {k: d[k] for k in keys if k in d}

def get_sub_dict(d, keys):
    '''获取字典的子集,仅仅是filter_dict的别名'''
    return filter_dict(d, keys)

def update_dict(original_dict, new_dict):
    '''更新字典,保护原始字典'''
    if original_dict is None:
        original_dict = {}
    if new_dict is None:
        return original_dict.copy()
    else:
        return {**original_dict, **new_dict}

def swap_dict_level(nested_dict, level_mapping):
    """
    交换嵌套字典的层顺序

    参数:
        nested_dict: 多层嵌套的字典
        level_mapping: 层映射字典,格式为 {原层位置: 新层位置}
                      例如 {0: 2, 1: 0, 2: 1} 表示原来的第0层移到第2层,第1层移到第0层,第2层移到第1层

    返回:
        重新排列层顺序后的新字典
    """
    def _validate_level_mapping(level_mapping):
        for k, v in level_mapping.items():
            if v not in level_mapping.keys():
                raise ValueError(f"Invalid level mapping: {level_mapping}. Each new position must correspond to an existing original position.")

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

    # 确定最大深度
    max_depth = len(next(iter(flat_dict.keys())))

    # 创建完整的重排顺序
    new_order = list(range(max_depth))
    for old_pos, new_pos in level_mapping.items():
        if old_pos < max_depth and new_pos < max_depth:
            new_order[new_pos] = old_pos

    rearranged_flat = {}
    for path, value in flat_dict.items():
        new_path = tuple(path[new_order[i]] for i in range(max_depth))
        rearranged_flat[new_path] = value

    return _nest_dict(rearranged_flat)

class BuildKeyStrategy(abc.ABC):
    """生成键的通用策略基类"""
    def __init__(self, included_name_list=None):
        if included_name_list is None:
            included_name_list = []
        self.included_name_list = included_name_list

    def validate_params(self, kwargs, allowed_params=None):
        if not kwargs:
            raise ValueError("至少需要提供一个有效参数")

        if allowed_params is not None:
            invalid = [p for p in kwargs if p not in allowed_params]
            if invalid:
                raise ValueError(f"无效参数: {invalid},允许的参数为: {allowed_params}")

    def _append_to_parts(self, parts, name, value):
        if name in self.included_name_list:
            parts.append(name)
        parts.append(value)

    @abc.abstractmethod
    def build_key(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        """直接调用时触发build_key"""
        return self.build_key(**kwargs)

class SortedBuildKey(BuildKeyStrategy):
    """按参数名首字母顺序生成键"""
    def build_key(self, **kwargs):
        self.validate_params(kwargs)
        sorted_params = sorted(kwargs.keys())
        parts = []
        for name in sorted_params:
            value = kwargs[name]
            self._append_to_parts(parts, name, value)
        return cat(*parts)

class OrderedBuildKey(BuildKeyStrategy):
    """按预定义顺序生成键"""
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
    """
    数据容器,内部是字典,但是可以使用kwargs访问value
    自动按参数名首字母顺序生成键的数据容器,无需指定param_order
    """
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
        '''
        按照各类dict的函数读取即可
        '''
        pass

class OrderedDataContainer(DataContainer):
    '''
    数据容器,内部是字典,但是可以使用kwargs访问value

    使用示例:
    dc = DataContainer(param_order=['area', 'prop', 'time_period'])
    dc.set_value(area='V1', prop='spike', time_period='pre_stimulus_period', value=1)
    dc.get_value(area='V1', prop='spike', time_period='pre_stimulus_period')  # 返回 1

    # 可以不输入完全的参数
    dc.set_value(area='V1', prop='spike', value=1)
    dc.get_value(area='V1', prop='spike')  # 返回 1

    # 可以从另一个DataContainer中更新数据
    dc2 = DataContainer(param_order=['area', 'prop', 'time_period'])
    dc2.set_value(area='V1', prop='spike', time_period='stimulus_period', value=2)
    dc.absorb(dc2)

    # 可以在吸收时添加新的参数
    dc.absorb_with_new_params(dc2, new_param='new_value')

    # 可以从字典中吸收数据
    dc.absorb_from_dict()

    # 可以获取子数据容器
    subcontainer = dc.get_subcontainer(area='V1', prop='spike')
    '''
    def __init__(self, param_order, included_name_list=None):
        super().__init__(included_name_list=included_name_list)
        self.key_builder = OrderedBuildKey(param_order=param_order, included_name_list=included_name_list)
        self['_config']['param_order'] = param_order

def _dict_to_data_container(d):
    '''
    将字典转换为DataContainer对象(需要自带_config和其中的param_map)

    示例:
    读取了一个字典,并将其转换为DataContainer对象
    '''
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
    '''
    展开嵌套列表(要求每一层都是列表)

    参数：
    - input_list: 嵌套列表
    - level: 展开的层级数,如果为1,则从外往内展开一层,如果为2,则从外往内展开两层,以此类推;如果为None,则展开所有层级
    '''
    if level is None:
        level = float('inf')  # 默认情况下展开所有层

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
    '''
    以original为模板,根据flattened中的元素重新构建一个列表.其括号结构与original相同,但元素顺序与flattened相同.这个函数必须要输出index才能不断调用自己,一般而言建议使用rebuild_list作为外部的接口.
    '''
    result = []
    for item in original:
        if isinstance(item, list):
            sub_list, index = rebuild_list_with_index(flattened, item, index)
            result.append(sub_list)
        else:
            result.append(flattened[index])
            index += 1
    return result, index

def rebuild_list(flattened, original):
    '''以original为模板,根据flattened中的元素重新构建一个列表.其括号结构与original相同,但元素顺序与flattened相同.'''
    return rebuild_list_with_index(flattened, original)[0]

def pure_list(l):
    '''遍历嵌套列表,将所有数组转换为列表,但保持嵌套结构不变.'''
    if isinstance(l, list):
        # 如果是列表,递归地对每个元素调用 pure_list
        return [pure_list(item) for item in l]
    elif isinstance(l, np.ndarray):
        # 如果是numpy数组,且其形状大于1(即不是单个数字),先将其转换为列表
        if l.shape:  # 检查数组是否不仅仅是单个元素
            return [pure_list(item) for item in l.tolist()]
        else:
            # 对于单个元素的numpy数组,直接返回它的Python类型
            return l.item()
    elif isinstance(l, tuple):
        # 如果是元组,先将其转换为列表
        return [pure_list(item) for item in l]
    else:
        # 对于其他类型的元素,直接返回
        return l

def list_shape(lst):
    '''
    假定list是和numpy的array类似的嵌套列表,返回list的形状.(不能处理不规则的嵌套列表)
    '''
    if not isinstance(lst, list):
        return ()
    if len(lst) == 0:
        return (0,)
    return (len(lst),) + list_shape(lst[0])

def is_sublist(sub_list, main_list):
    '''
    判断一个列表是否是另一个列表的子集
    '''
    main_counter = Counter(main_list)
    sub_counter = Counter(sub_list)
    for elem, count in sub_counter.items():
        if main_counter[elem] < count:
            return False
    return True

class InstanceContainer:
    '''
    用来存储一组实例,并提供一些常用的操作方法,如过滤/提取信息等

    使用示例:
    container = InstanceContainer(items=[obj1, obj2, obj3])
    for item in container:
        print(item)
    print(len(container))
    print(container[0])
    '''
    def __init__(self, items=None):
        self._instances = list(items) if items else []

    def append(self, item):
        self._instances.append(item)

    def extend(self, items):
        self._instances.extend(items)

    def __len__(self):
        return len(self._instances)

    def __iter__(self):
        return iter(self._instances)

    def __getitem__(self, index):
        return self._instances[index]

def reorder_df(df, new_row=None, new_col=None):
    '''
    按照指定顺序对DataFrame的行和列进行排序
    '''
    if new_row is not None:
        df = df.loc[new_row]
    if new_col is not None:
        df = df[new_col]
    return df

def align_df_row_col_order(df, keep_which='row'):
    '''
    确保df的行和列顺序一致
    '''
    if keep_which == 'row':
        return df.sort_index(axis=1).sort_index(axis=0)
    elif keep_which == 'col':
        return df.sort_index(axis=0).sort_index(axis=1)

def get_sub_df(df, row_list=None, col_list=None):
    '''
    获取DataFrame的子集
    '''
    if row_list is not None:
        new_df = df.loc[row_list]
    else:
        new_df = df.copy()
    if col_list is not None:
        new_df = new_df[col_list]
    return new_df

def replace_df_diagonal(df, replace_value=0):
    '''
    将DataFrame对角线上的元素替换为指定值
    '''
    df_copy = df.copy()
    for row in df_copy.index:
        for col in df_copy.columns:
            if row == col:
                df_copy.loc[row, col] = replace_value
    return df_copy

def get_zero_df(index, columns):
    '''创建一个DataFrame'''
    df = pd.DataFrame(np.zeros((len(index), len(columns))), index=index, columns=columns)
    index_col_str(df)
    return df

def index_col_str(df):
    '''将df的index和columns转换为字符串'''
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

def normalize_df_with_sum(df, row_or_col):
    '''
    正则化df,使得行或列的和为1

    注意:
    nan值会被当作0处理
    '''
    if row_or_col == 'row':
        axis = 1
    elif row_or_col == 'col':
        axis = 0

    return df.div(df.sum(axis=axis), axis=1-axis)

def row_sum_df(df):
    '''计算DataFrame每行的和'''
    return df.sum(axis=1)

def col_sum_df(df):
    '''计算DataFrame每列的和'''
    return df.sum(axis=0)

def insert_mid(arr):
    '''在数组中间插入相邻元素的平均值'''
    new_arr = []
    for i in range(len(arr)):
        new_arr.append(arr[i])
        if i < len(arr) - 1:
            new_arr.append((arr[i] + arr[i+1]) / 2)

    if isinstance(arr, np.ndarray):
        return np.array(new_arr)
    else:
        return new_arr

def sort_dict(d, sort_by='value', reverse=False):
    '''
    Sort a dictionary by its keys or values.

    Parameters:
    - d: The dictionary to sort.
    - sort_by: Criteria for sorting ('value' or 'key').
    - reverse: Sort in descending order if True, else in ascending order.

    Returns:
    - A new sorted dictionary.
    '''
    if sort_by == 'key':
        return {k: d[k] for k in sorted(d.keys(), reverse=reverse)}
    elif sort_by == 'value':
        return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}
    else:
        raise ValueError("sort_by must be 'key' or 'value'")

def get_sorted_index_from_sorted_list(l, sorted_l):
    index_map = {}
    for i, val in enumerate(l):
        if val not in index_map:
            index_map[val] = []
        index_map[val].append(i)

    result = []
    for val in sorted_l:
        result.append(index_map[val].pop(0))
    return result

def sort_list_using_index(l, indices):
    new_l = [0] * len(l)
    for i, idx in enumerate(indices):
        new_l[i] = l[idx]
    return new_l

def get_default_param(func):
    sig = inspect.signature(func)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def get_param_value(func, args, kwargs, param_name):
    """
    获取函数指定参数的值,包括考虑默认值的情况.(适合在decorator内使用,因为需要获取某个变量的值并根据这个变量做一些操作)

    :param func: 被装饰的函数
    :param args: 位置参数
    :param kwargs: 关键字参数
    :param param_name: 需要获取值的参数名称
    :return: 参数的值(如果存在),否则为 None
    """
    signature = inspect.signature(func)
    bound_args = signature.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()  # 应用默认值

    # 获取指定参数的值
    return bound_args.arguments.get(param_name, None)

def add_method_to_instance(instance, method_func, method_name):
    """
    将方法动态添加到一个实例中.

    参数:
        instance: 目标实例
        method_func: 要添加的函数,需包含self参数
        method_name: 添加后的方法名称
    """
    bound_method = types.MethodType(method_func, instance)
    setattr(instance, method_name, bound_method)

def process_inf(data, inf_policy=INF_POLICY, verbose=False):
    '''
    Processes inf values according to the specified policy.

    When ``verbose`` is True, a message is printed only if one or more infinite
    values are actually converted to NaN.
    '''
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
            print(
                f"process_inf: processed {inf_count} Inf value(s) "
                f"with inf_policy='{inf_policy}'."
            )
    elif inf_policy == 'ignore':
        pass
    return data

def process_nan(data, nan_policy=NAN_POLICY, fill_value=None, verbose=False):
    '''
    主要适用于一维数据,对于二维数据,drop interplate等操作不被允许,需要使用者分解为一维数据后再进行操作

    参数:
    - data: 输入的数据
    - nan_policy: 可选'propagate', 'fill', 'drop', 'interpolate'之一,默认为NAN_POLICY; propogate表示不处理NaN,fill表示用fill_value填充NaN,drop表示删除NaN,interpolate表示使用插值填充NaN
    - fill_value: 用于填充NaN的值;仅在nan_policy='fill'时需要
    - verbose: 为True时,仅在实际处理了NaN后打印处理数量和策略
    '''
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
            result = df_like_interpolate(data)  # 使用和Pandas一致的插值方法
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
            result = df_like_interpolate(data)  # 使用和Pandas一致的插值方法
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
            result = data.interpolate()  # 线性插值填充NaN
        else:
            raise ValueError('nan_policy not supported')

    else:
        return None

    nan_count_after = int(np.count_nonzero(np.asarray(pd.isna(result))))
    processed_count = nan_count_before - nan_count_after
    if verbose and processed_count:
        print(
            f"process_nan: processed {processed_count} NaN value(s) "
            f"with nan_policy='{nan_policy}'."
        )
    return result

def process_special_value(data, nan_policy=NAN_POLICY, fill_value=None, inf_policy=INF_POLICY, verbose=False):
    '''
    处理数据中的NaN和inf值.

    ``verbose=True``时,仅由基础处理函数报告实际发生的修改;本函数不重复汇总.
    '''
    data = process_inf(data, inf_policy=inf_policy, verbose=verbose)  # Process inf values first
    data = process_nan(data, nan_policy=nan_policy,
                       fill_value=fill_value, verbose=verbose)  # Process NaN values
    return data

def sync_special_value(*args, inf_policy=INF_POLICY, verbose=False):
    '''
    同步多个输入数据中相同位置的NaN和inf值.

    所有输入必须类型和形状一致,支持list、numpy.ndarray、pd.Series和
    pd.DataFrame。Series还必须具有相同索引,DataFrame必须具有相同索引和列.
    本函数按位置同步特殊值,不会修改原始输入.
    ``verbose=True``时报告Inf转换和因同步新增的NaN单元数量.
    '''
    if not args:
        raise ValueError('At least one input is required')

    supported_types = (list, np.ndarray, pd.Series, pd.DataFrame)
    if not all(isinstance(a, supported_types) for a in args):
        raise TypeError('All inputs must be lists, numpy arrays, pandas Series, or pandas DataFrames')

    input_type = type(args[0])
    if any(type(a) is not input_type for a in args[1:]):
        raise TypeError('All inputs must have the same type')

    # 统一通过numpy获取形状,从而正确检查一维和多维list
    shapes = [np.asarray(a).shape for a in args]
    if len(set(shapes)) > 1:
        raise ValueError("All inputs must have the same shape")

    if input_type is pd.Series:
        reference_index = args[0].index
        if any(not a.index.equals(reference_index) for a in args[1:]):
            raise ValueError('All pandas Series inputs must have the same index')
    elif input_type is pd.DataFrame:
        reference_index = args[0].index
        reference_columns = args[0].columns
        if any(
            not a.index.equals(reference_index) or not a.columns.equals(reference_columns)
            for a in args[1:]
        ):
            raise ValueError('All pandas DataFrame inputs must have the same index and columns')

    args = [process_inf(a, inf_policy=inf_policy, verbose=verbose)
            for a in args]  # Process inf values first

    # Convert all inputs to numpy arrays for processing
    args_np = [np.array(a) if not isinstance(
        a, np.ndarray) else a for a in args]

    # Find indices where any input has NaN
    nan_mask = np.any(np.isnan(args_np), axis=0)
    synchronized_count = sum(
        int(np.count_nonzero(nan_mask & ~np.isnan(arg)))
        for arg in args_np
    )

    # Apply NaN where any NaNs are found across corresponding elements
    args_np_synced = [np.where(nan_mask, np.nan, arg) for arg in args_np]
    if verbose and synchronized_count:
        print(
            f"sync_special_value: synchronized {synchronized_count} "
            "value(s) to NaN."
        )

    # Convert back to original data types
    args_synced = []
    for original_arg, synced_arg in zip(args, args_np_synced):
        if isinstance(original_arg, list):
            args_synced.append(synced_arg.tolist())
        elif isinstance(original_arg, pd.Series):
            args_synced.append(pd.Series(synced_arg, index=original_arg.index))
        elif isinstance(original_arg, pd.DataFrame):
            args_synced.append(pd.DataFrame(
                synced_arg, index=original_arg.index, columns=original_arg.columns))
        elif isinstance(original_arg, np.ndarray):
            args_synced.append(synced_arg)

    return args_synced if len(args_synced) > 1 else args_synced[0]

def has_invalid_value(value):
    if isinstance(value, dict):
        return any(has_invalid_value(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_invalid_value(item) for item in value)
    array = np.asarray(value)
    if array.dtype == object and array.ndim:
        return any(has_invalid_value(item) for item in array.flat)
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
    '''
    将数据缩放到新的范围.

    参数:
    - data: 输入数据
    - old_min: 原始数据的最小值.
    - old_max: 原始数据的最大值.
    - new_min: 新的最小值.
    - new_max: 新的最大值.
    '''
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def lowess_smooth(x, y, frac=0.2):
    """
    对给定的 x 和 y 数据使用 LOWESS 进行平滑处理.函数内部会确保 x 是有序的.

    参数:
    x (array-like): 自变量(输入)的数据.
    y (array-like): 因变量(输出)的数据.
    frac (float): 控制局部回归时使用的窗口大小,值越大,曲线越平滑.默认值为 0.2.

    返回:
    x_smooth (numpy array): 平滑后的 x 数据.
    y_smooth (numpy array): 平滑后的 y 数据.
    """
    # 确保是array
    x = np.array(x)
    y = np.array(y)

    # 确保 x 是有序的,首先对 x 和 y 进行排序
    sort_index = np.argsort(x)
    x_sorted = x[sort_index]
    y_sorted = y[sort_index]

    # 使用 LOWESS 进行平滑
    lowess_result = lowess(y_sorted, x_sorted, frac=frac)

    # 从结果中提取平滑后的 x 和 y 值
    x_smooth = lowess_result[:, 0]
    y_smooth = lowess_result[:, 1]

    return x_smooth, y_smooth

def get_angle(source, target, angle_type='rad', angle_range='0:360'):
    '''
    计算source到target的向量的角度

    参数:
    source - 起始点(2D),支持同时输入多个点,形状为(N, 2)或者(2,)
    target - 终点(2D),支持同时输入多个点,形状为(N, 2)或者(2,)
    angle_type - 角度类型,支持'rad'和'deg'
    angle_range - 角度范围,支持'0:360'和'-180:180'

    返回:
    angle - 角度,如果输入是多个点,返回形状为(N,),如果输入是单个点,返回标量
    '''
    source = np.array(source)
    target = np.array(target)
    if len(source.shape) == 1:
        source = source.reshape(1, -1)
    if len(target.shape) == 1:
        target = target.reshape(1, -1)

    if source.shape[0] != target.shape[0] and source.shape[0] != 1 and target.shape[0] != 1:
        raise ValueError("The number of source and target points must be the same.")

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
    '''
    计算source到target的向量的水平角(azimuth)和垂直角(elevation)

    参数:
    source - 起始点(3D),支持同时输入多个点,形状为(N, 3)或者(3,)
    target - 终点(3D),支持同时输入多个点,形状为(N, 3)或者(3,)
    angle_type - 角度类型,支持'rad'和'deg'
    azim_range - 水平角范围,支持'0:360'和'-180:180'
    elev_range - 垂直角范围,支持'0:180'和'-90:90';
                 '-90:90'表示相对x-y平面的仰角(向上为正),
                 '0:180'表示从+z轴向下量取的天顶角

    返回:
    azim - 水平角度,如果输入是多个点,返回形状为(N,),如果输入是单个点,返回标量
    elev - 垂直角度,如果输入是多个点,返回形状为(N,),如果输入是单个点,返回标量
    '''
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

    if source.shape[0] != target.shape[0] and source.shape[0] != 1 and target.shape[0] != 1:
        raise ValueError("The number of source and target points must be the same.")

    # 计算水平角(azimuth)和垂直角(elevation)
    dx = target[:, 0] - source[:, 0]
    dy = target[:, 1] - source[:, 1]
    dz = target[:, 2] - source[:, 2]
    azim = np.arctan2(dy, dx)
    elev = np.arctan2(dz, np.sqrt(dx**2 + dy**2))

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

    return azim, elev

def get_corr(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, sync=True, verbose=False):
    '''
    计算两个数组的相关系数
    sync表示是否将两个数组的nan和inf位置同步处理(假如一个数据有nan,另一个没有,则将两个数据的nan位置都设置为nan)
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy, verbose=verbose) if sync else (x, y)
    return np.corrcoef(process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose), process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose))[0, 1]

def get_rectangle_matrix_corr(matrix_x, matrix_y, diagonal=True, get_corr_kwargs=None):
    '''
    计算两个矩阵flatten后的相关系数,非方阵
    diagonal: True or False,表示是否计算对角线上的相关系数,默认为True,表示计入对角线
    get_corr_kwargs: dict, get_corr函数的参数
    '''
    if get_corr_kwargs is None:
        get_corr_kwargs = {}

    if diagonal:
        matrix_x_flat = matrix_x.flatten()
        matrix_y_flat = matrix_y.flatten()
    else:
        # 创建非对角线元素的掩码
        rows, cols = matrix_x.shape
        mask = np.eye(rows, cols, dtype=bool)
        non_diag_mask = ~mask

        # 应用掩码并展平
        matrix_x_flat = matrix_x[non_diag_mask].flatten()
        matrix_y_flat = matrix_y[non_diag_mask].flatten()

    # 调用相关系数计算函数
    return get_corr(matrix_x_flat, matrix_y_flat, **get_corr_kwargs)

def get_pd_rectangle_corr(df_x, df_y, diagonal=True, get_corr_kwargs=None):
    '''
    计算两个DataFrame的相关系数,类似于get_rectangle_matrix_corr
    '''
    df_x_aligned = align_df_row_col_order(df_x)
    df_y_aligned = align_df_row_col_order(df_y)
    return get_rectangle_matrix_corr(df_x_aligned.values, df_y_aligned.values, diagonal=diagonal, get_corr_kwargs=get_corr_kwargs)

def get_linregress(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, sync=True, verbose=False):
    '''
    对给定的数据进行线性回归拟合

    参数:
    x - x数据
    y - y数据

    返回:
    slope - 斜率
    intercept - 截距
    r - 相关系数
    p - p值
    std_err - 标准误差
    residual - 残差(y - y_pred)
    '''
    x, y = sync_special_value(x, y, inf_policy=inf_policy, verbose=verbose) if sync else (x, y)
    x = np.array(process_special_value(x, nan_policy=nan_policy,
                 fill_value=fill_value, inf_policy=inf_policy, verbose=verbose))
    y = np.array(process_special_value(y, nan_policy=nan_policy,
                 fill_value=fill_value, inf_policy=inf_policy, verbose=verbose))

    regress_dict = {}

    # 计算线性回归
    regress_dict['slope'], regress_dict['intercept'], regress_dict['r'], regress_dict['p'], regress_dict['std_err'] = st.linregress(
        x, y)
    regress_dict['residual'] = y - \
        (x * regress_dict['slope'] + regress_dict['intercept'])

    return regress_dict

def get_ks(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False):
    '''
    计算两个数组的Kolmogorov-Smirnov距离

    参数:
    x - 第一个数组
    y - 第二个数组或者CDF函数

    返回:
    ks_distance - 两个数组的KS距离
    '''
    return get_ks_result(x, y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)['ks']

def get_ks_result(x, y, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False):
    '''
    计算两个数组的Kolmogorov-Smirnov距离及其P值,注意当分布一样时,KS距离为0,而P值为1

    参数:
    x - 第一个数组
    y - 第二个数组或者CDF函数

    返回:
    ks_distance - 两个数组的KS距离
    p_value - 对应的P值
    '''
    # 处理输入x的特殊值
    x_clean = process_special_value(x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)

    if callable(y):
        # 单样本检验：y是CDF函数
        ks_result = st.kstest(x_clean, y)
    else:
        # 双样本检验：处理y的特殊值后计算
        y_clean = process_special_value(y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
        ks_result = st.ks_2samp(x_clean, y_clean)

    r = {'ks': ks_result.statistic, 'p': ks_result.pvalue}
    return r

def get_kl_divergence(p, q, base=None, **kwargs):
    '''
    计算两个概率分布的KL散度
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布

    返回:
    kl_divergence - 两个概率分布的KL散度
    '''
    return st.entropy(p, q, base=base, **kwargs)

def get_jsd(p, q, base=None, **kwargs):
    '''
    计算两个概率分布的Jensen-Shannon散度
    两个概率分布的长度必须相同

    参数:
    p - 第一个概率分布
    q - 第二个概率分布

    返回:
    jsd - 两个概率分布的Jensen-Shannon散度
    '''
    m = (np.array(p) + np.array(q)) / 2
    return (get_kl_divergence(p, m, base=base, **kwargs) + get_kl_divergence(q, m, base=base, **kwargs)) / 2

def get_hist(data, bins, stat='probability', nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False):
    '''
    Generates a histogram with various normalization options.

    Parameters:
    - data: array-like, the data to be histogrammed.
    - bins: int or array-like, the bin specification.
    - stat: str, the normalization method to use. One of ['count', 'frequency', 'probability', 'proportion', 'percent', 'density'] as sns.histplot.

    Returns:
    - values: array, the values corresponding to the normalization method chosen.
    - bin_edges: array, the edges of the bins.
    - midpoints: array, the midpoints of the bins.
    '''

    local_data = process_special_value(data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)

    hist, bin_edges = np.histogram(local_data, bins=bins)

    # Calculate bin widths
    bin_widths = np.diff(bin_edges)

    if stat == 'count':
        # No change needed, hist already represents the count
        values = hist
    elif stat == 'frequency':
        # Frequency is count divided by bin width
        values = hist / bin_widths
    elif stat == 'probability' or stat == 'proportion':
        # Probability is count divided by total number of observations
        values = hist / hist.sum()
    elif stat == 'percent':
        # Percent is probability times 100
        values = (hist / hist.sum()) * 100
    elif stat == 'density':
        # Density normalizes count by total observations and bin width
        values = hist / (hist.sum() * bin_widths)
    else:
        raise ValueError("Unsupported stat method.")

    return values, bin_edges, get_midpoint(bin_edges)

def get_hist_2d(data_x, data_y, bins_x, bins_y, stat='probability', nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, sync=True, verbose=False):
    '''
    Generates a 2D histogram with various normalization options for two-dimensional data,
    allowing separate bin specifications for each axis.

    Parameters:
    - data_x: array-like, the data for the x-axis to be histogrammed.
    - data_y: array-like, the data for the y-axis to be histogrammed.
    - bins_x: int or array-like, the bin specification for the x-axis.
    - bins_y: int or array-like, the bin specification for the y-axis.
    - stat: str, the normalization method to use. One of ['count', 'frequency', 'probability', 'proportion', 'percent', 'density'].

    Returns:
    - values: 2D array, the values corresponding to the normalization method chosen.
    - bin_edges_x: array, the edges of the bins for the x-axis.
    - bin_edges_y: array, the edges of the bins for the y-axis.
    - midpoints_x: array, the midpoints of the bins for the x-axis.
    - midpoints_y: array, the midpoints of the bins for the y-axis.
    '''

    if sync:
        local_data_x, local_data_y = sync_special_value(data_x, data_y, inf_policy=inf_policy, verbose=verbose)
    else:
        local_data_x, local_data_y = data_x.copy(), data_y.copy()
    local_data_x = process_special_value(local_data_x, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    local_data_y = process_special_value(local_data_y, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)

    hist, bin_edges_x, bin_edges_y = np.histogram2d(
        local_data_x, local_data_y, bins=[bins_x, bins_y])

    # Calculate bin areas for density calculation
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
        values = (hist / hist.sum()) * 100
    elif stat == 'density':
        values = hist / (hist.sum() * bin_areas)
    else:
        raise ValueError("Unsupported stat method.")

    return values, bin_edges_x, bin_edges_y, get_midpoint(bin_edges_x), get_midpoint(bin_edges_y)

def get_cdf(data, nan_policy='drop', fill_value=None, inf_policy=INF_POLICY, verbose=False, **kwargs):
    '''
    计算累积分布函数 (CDF) 并返回结果
    :param data: 1-d array-like, the input data.
    '''
    clean_data = process_special_value(
        data, nan_policy=nan_policy, fill_value=fill_value, inf_policy=inf_policy, verbose=verbose)
    return ECDF(clean_data, **kwargs)

def get_midpoint(x):
    '''
    Calculate the midpoints of a sequence of values.

    Parameters:
    - x: array-like, the values to calculate midpoints for.

    Returns:
    - midpoints: array, the midpoints of the input values.
    '''
    return (x[1:] + x[:-1]) / 2

def repeat_data(data, repeat_times):
    '''
    重复列表中的每个元素指定的次数.

    参数:
    - data: 要重复的数据,可以是列表,numpy数组,字典,Pandas Series或DataFrame
    - repeat_times: 重复的次数,可以是整数,列表,numpy数组,字典,Pandas Series或DataFrame

    返回:
    - 所有重复后的元素
    '''
    # For lists and numpy arrays
    if isinstance(data, list):
        if isinstance(repeat_times, int):
            return [item for item in data for _ in range(repeat_times)]
        elif isinstance(repeat_times, (list, np.ndarray)) and len(data) == len(repeat_times):
            return [item for item, count in zip(data, repeat_times) for _ in range(count)]
    elif isinstance(data, np.ndarray):
        if isinstance(repeat_times, (list, np.ndarray, int)):
            return np.repeat(data, repeat_times)

    # Handling dictionary data
    if isinstance(data, dict) and isinstance(repeat_times, (dict, int)):
        result = []
        if isinstance(repeat_times, int):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend(
                        [item for item in value for _ in range(repeat_times)])
                else:
                    result.extend([value for _ in range(repeat_times)])
        elif isinstance(repeat_times, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    result.extend(
                        [item for item in value for _ in range(repeat_times[key])])
                else:
                    result.extend(
                        [value for _ in range(repeat_times[key])])
        return result

    # Convert Series and DataFrame outputs to list or np.array using pandas.concat
    if isinstance(data, pd.Series):
        if isinstance(repeat_times, int):
            return data.repeat(repeat_times).tolist()
        elif isinstance(repeat_times, pd.Series):
            repeated_values = [pd.Series(
                [value] * repeat_times.get(index, 1)) for index, value in data.items()]
            repeated_series = pd.concat(repeated_values).reset_index(drop=True)
            return repeated_series.tolist()
    if isinstance(data, pd.DataFrame):
        if isinstance(repeat_times, int):
            return data.loc[data.index.repeat(repeat_times)].to_numpy()
        elif isinstance(repeat_times, pd.DataFrame):
            data_flat = data.to_numpy().flatten()
            repeats_flat = repeat_times.to_numpy().flatten().astype(int)
            return np.repeat(data_flat, repeats_flat)

class MetaResultsMixin:
    '''
    将某个f'{results_prefix}_results'的结果设置为特定的类型,并且灵活的保存和读取
    '''
    def _set_results(self, results_prefix, results_type='dict', param_order=None):
        '''
        设置结果的类型并初始化
        '''
        setattr(self, f'{results_prefix}_results_type', results_type)
        if results_type == 'dict':
            setattr(self, f'{results_prefix}_results', {})
        elif results_type == 'OrderedDataContainer':
            setattr(self, f'{results_prefix}_results', OrderedDataContainer(param_order=param_order))

    def _set_save_load_method(self, results_prefix, save_load_method='separate'):
        '''
        注意:
        这里对于separate使用了save_dict_separate_merge_to_saved,方便只增加部分新results
        '''
        # 验证保存加载方法
        acceptable_methods = ['separate', 'lmdb']
        if save_load_method not in acceptable_methods:
            raise ValueError(f'save_load_method must be one of {acceptable_methods}')

        # 设置基础属性名
        save_attr = f'save_{results_prefix}_func'
        load_attr = f'load_{results_prefix}_func'
        method_attr = f'{results_prefix}_save_load_method'

        # 设置方法类型
        setattr(self, method_attr, save_load_method)

        # 分配对应的函数
        if save_load_method == 'separate':
            setattr(self, save_attr, save_dict_separate_merge_to_saved)
            setattr(self, load_attr, load_dict_separate_merge_to_exist)
        elif save_load_method == 'lmdb':
            setattr(self, save_attr, save_dict_lmdb)
            setattr(self, load_attr, load_dict_lmdb_merge_to_exist)


    def _update_config_from_saved_results(self, results_prefix):
        '''
        从已经保存的结果中更新_config
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        if check_all_file_exist(pj(self.outcomes_dir, f'{results_prefix}_results')):
            try:
                load_func = getattr(self, f'load_{results_prefix}_func')
                load_func(results_dict, pj(self.outcomes_dir, f'{results_prefix}_results'), key_to_load=['_config'], ensure_config=False)
            except Exception as e:
                # print(f'Error loading config from {results_prefix}_results: {e}')
                pass
        self.__setattr__(f'{results_attr}_config_updated', True)

    def _get_results_value_by_key(self, results_prefix, key):
        '''
        从结果中获取值,但是只支持直接输入key

        此实现方式避免了一次性读取所有分析结果,节省内存
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        if key in results_dict:
            return results_dict[key]
        else:
            load_func = getattr(self, f'load_{results_prefix}_func')
            partial_results = load_func(results_dict, pj(self.outcomes_dir, f'{results_prefix}_results'), key_to_load=[key], ensure_config=False)
            return results_dict[key]

    def _get_results_value(self, results_prefix, key=None, **kwargs):
        '''
        兼容dict和datacontainer的情况
        '''
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        results_type = getattr(self, f'{results_prefix}_results_type')

        if results_type == 'dict':
            return self._get_results_value_by_key(results_prefix=results_prefix, key=key)
        elif results_type == 'OrderedDataContainer':
            if key is None:
                key = results_dict.build_key(**kwargs)
            return self._get_results_value_by_key(results_prefix=results_prefix, key=key)


    def _save_results(self, results_prefix, filename, **kwargs):
        results_attr = f'{results_prefix}_results'
        results_dict = getattr(self, results_attr)
        filename = pj(self.outcomes_dir, filename)
        getattr(self, f'save_{results_prefix}_func')(results_dict, filename, **kwargs)
        print_title(f'{self.model_name}: {results_prefix} results saved')

    def _mark_all_results_saved(self, results_prefix):
        '''
        标志着所有结果都已经保存完毕
        '''
        save_dict({f'{results_prefix}_results_saved': True}, pj(self.outcomes_dir, f'{results_prefix}_results', f'{results_prefix}_results_saved'))

    def _check_all_results_saved(self, results_prefix):
        '''
        确认是否所有结果都已经保存完毕
        '''
        if check_all_file_exist_with_any_extension(pj(self.outcomes_dir, f'{results_prefix}_results', f'{results_prefix}_results_saved')):
            setattr(self, f'{results_prefix}_results_saved', True)
            return True
        else:
            setattr(self, f'{results_prefix}_results_saved', False)
            return False

class MetaAnalyzerMixin(MetaResultsMixin):


    def set_analysis_task(self, analysis_task_list):
        '''
        设置分析结果的task列表
        '''
        self.analysis_task_list = analysis_task_list


    def set_analysis_results(self, results_type='dict', param_order=None):
        self._set_results(results_prefix='analysis', results_type=results_type, param_order=param_order)

    def set_analysis_save_load_method(self, save_load_method='separate'):
        self._set_save_load_method(results_prefix='analysis', save_load_method=save_load_method)

class MetaModel(abc.ABC, MetaAnalyzerMixin):
    '''
    一个元模型的基类,用于封装模型的参数设置,读取,保存;cpu,gpu设置;备份代码;设置logger等功能

    注意:
    当模型分为多个阶段,下一阶段依赖上一阶段结果时,可以嵌套Metamodel
    这么做的好处是,上一阶段在不被修改的时候不会被频繁运行,而是可以读取到已有结果
    例如,SNN生成连接可以单独作为一个model,运行snn也是一个model,运行SNN的model内部使用connection的model

    警告:
    当模型运算速度非常快,并且串行跑了多个,有可能导致timedir是一样的,这将导致后面的模型覆盖前面的模型
    当使用multiprocess运行时也有可能发生这种情况,一般不会,但是记录在此
    '''
    def __init__(self):
        self.force_run = False
        self.set_simulation_results() # 直接结果(simulation得到的),如果需要可以在子类改成DataContainer
        self.set_analysis_results() # 分析结果(analysis得到的),如果需要可以在子类改成DataContainer
        self.pop = False
        self.value_dir_key_before = []
        self.both_dir_key_before = []
        self.value_dir_key_after = []
        self.both_dir_key_after = []
        self.ignore_key_list = []
        self.code_file_list = []
        self.set_current_time()
        self.set_model_name()
        self.set_optional_params_default()

    # region set things
    def set_up(self, params, basedir, code_file_list, value_dir_key_before=None, both_dir_key_before=None, value_dir_key_after=None, both_dir_key_after=None, ignore_key_list=None, force_run=None):
        '''
        总体设置

        注意:
        不输入的参数会使用默认值
        '''
        if value_dir_key_before is None:
            value_dir_key_before = self.value_dir_key_before
        if both_dir_key_before is None:
            both_dir_key_before = self.both_dir_key_before
        if value_dir_key_after is None:
            value_dir_key_after = self.value_dir_key_after
        if both_dir_key_after is None:
            both_dir_key_after = self.both_dir_key_after
        if ignore_key_list is None:
            ignore_key_list = self.ignore_key_list
        if force_run is None:
            force_run = self.force_run
        self.set_params(params=params)
        self.set_basedir(basedir=basedir)
        self.set_value_dir_both_dir_key(value_dir_key_before=value_dir_key_before, both_dir_key_before=both_dir_key_before, value_dir_key_after=value_dir_key_after, both_dir_key_after=both_dir_key_after)
        self.set_ignore_key_for_search_params(ignore_key_list=ignore_key_list)
        self.create_timedir()
        self.set_code_file_list(code_file_list=code_file_list)
        self.set_force_run(force_run=force_run)

    def set_optional_params_default(self):
        self.set_analysis_task(['all'])
        self.set_simulation_save_load_method(save_load_method='separate')
        self.set_analysis_save_load_method(save_load_method='separate')


    def set_force_run(self, force_run=False):
        '''
        设置模型的强制运行标志,默认为False,会尝试读取已有结果,如果为True,则无条件运行模型
        '''
        self.force_run = force_run

    def set_basedir(self, basedir):
        '''
        设置模型的基础文件夹

        示例:
        basedir = '../../results'
        basedir = os.path.join(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
        这样可以根据代码文件名自动创建文件夹

        注意:
        basedir和timedir是不同的
        '''
        self.basedir = basedir

    def set_timedir(self, timedir):
        '''
        设置模型的时间文件夹

        示例:
        当已经手动确定了模型的时间文件夹时,可以直接设置,方便之后运行analysis

        注意:
        只有在希望单独运行analysis时才需要设置
        timedir不一定以时间结尾,特殊情况下时间后面还会有value_dir_key_after,both_dir_key_after等产生的文件夹
        '''
        self.timedir = timedir
        self.load_params()
        self.load_info_container()
        self._update_config_from_saved_results('simulation')
        self._update_config_from_saved_results('analysis')

    def set_current_time(self, current_time=None):
        '''
        设置当前时间,如果不设置,则使用当前时间
        '''
        if current_time is None:
            self.current_time = get_time()
        else:
            self.current_time = current_time

    def set_model_name(self, model_name=None):
        if model_name is None:
            self.model_name = self.__class__.__name__
        else:
            self.model_name = model_name

    def set_value_dir_both_dir_key(self, value_dir_key_before=None, both_dir_key_before=None, value_dir_key_after=None, both_dir_key_after=None):
        '''
        设置模型的value_dir_key和both_dir_key

        注意:
        输入none则不更新
        '''
        if value_dir_key_before is not None:
            self.value_dir_key_before = value_dir_key_before
        if both_dir_key_before is not None:
            self.both_dir_key_before = both_dir_key_before
        if value_dir_key_after is not None:
            self.value_dir_key_after = value_dir_key_after
        if both_dir_key_after is not None:
            self.both_dir_key_after = both_dir_key_after

    def set_ignore_key_for_search_params(self, ignore_key_list=None):
        '''
        设置搜索参数时忽略的key

        注意:
        输入none则不更新
        '''
        if ignore_key_list is not None:
            self.ignore_key_list = ignore_key_list


    def set_params(self, params, set_to_attr=True):
        '''
        设置模型参数
        '''
        if set_to_attr:
            for key, value in params.items():
                setattr(self, key, value)
        self.params = params

    def set_code_file_list(self, code_file_list):
        '''
        设置模型代码文件列表

        示例:
        code_file_list = [cf.current_file(), '../../utils_function/utils_function.py']
        '''
        self.code_file_list = code_file_list

    def set_logger(self, filename=None, filemode='a'):
        '''
        设置logger
        '''
        self.logger = Logger()
        self.logger.get_py_logger(basedir=self.logs_dir, filename=filename, filemode=filemode, name=self.timedir)
        print_title(f'{self.model_name}: Logger initialized')

    def set_simulation_results(self, results_type='dict', param_order=None):
        '''
        设置模拟结果的类型并初始化
        '''
        self._set_results(results_prefix='simulation', results_type=results_type, param_order=param_order)

    def set_simulation_save_load_method(self, save_load_method='separate'):
        self._set_save_load_method(results_prefix='simulation', save_load_method=save_load_method)

    # endregion

    # region run
    @property
    def should_run(self):
        """
        判断是否需要执行run_detail的逻辑聚合
        """
        if not self.params_exist:
            # 当参数不存在时,强制标记结果不存在
            self.simulation_results_exist = False
            return True
        return not self.simulation_results_exist

    def run(self):
        '''
        运行模型
        '''
        self.prepare_run()

        # 通过属性判断执行
        if self.should_run:
            initialize_timer = Timer(title=f'{self.model_name}: Initialize model')
            initialize_timer.start()
            self.initialize_model()
            initialize_timer.end()

            run_timer = Timer(title=f'{self.model_name}: Run model')
            run_timer.start()
            self.run_detail()
            run_timer.end()

            self.finalize_run()


    def prepare_run(self):
        '''
        run_detail的准备工作,包括search_params,check_simulation_results_exist,set_logger等
        '''
        self.search_params()

        # 自动触发结果存在性检查(当参数存在时)
        if self.params_exist:
            self.check_simulation_results_exist()

        # 统一设置日志(此时timedir已确定)
        self.set_logger()
        self.save_code()

    def initialize_model(self):
        '''
        部分模型需要在run前初始化,例如brainpy需要先构建net,monitor,runner才可以run
        '''
        self.save_info_container()
        self.save_params()

    @abc.abstractmethod
    def run_detail(self):
        """
        子类必须实现核心逻辑,需要将结果更新到self.simulation_results
        """
        pass

    def finalize_run_detail(self):
        '''
        子类可以更改
        '''
        if not self.simulation_results_exist:
            self.save_simulation_results()

    def finalize_run(self):
        """
        run的收尾操作,子类最好不要修改,而是修改finalize_run_detail
        """
        self.finalize_run_detail()
        self._mark_all_results_saved('simulation')
    # endregion

    # region visualize
    # endregion

    # region dir
    def create_timedir(self):
        '''
        获取模型的时间文件夹
        '''
        _, self.before_timedir = pop_dict_get_dir(self.params, self.value_dir_key_before, self.both_dir_key_before, self.basedir)
        _, self.after_timedir = pop_dict_get_dir(self.params, self.value_dir_key_after, self.both_dir_key_after, '')
        self.timedir = pj(self.before_timedir, self.current_time, self.after_timedir)


    @property
    def logs_dir(self):
        return pj(self.timedir, 'logs')

    @property
    def figs_dir(self):
        return pj(self.timedir, 'figs')


    @property
    def outcomes_dir(self):
        return pj(self.timedir, 'outcomes')

    @property
    def params_dir(self):
        return pj(self.timedir, 'params')


    @property
    def code_dir(self):
        '''保留之前的没有加s的版本'''
        return pj(self.timedir, 'code')
    # endregion

    # region load
    def search_params(self):
        '''
        搜索运行过相同参数的文件夹
        '''
        if self.force_run:
            print_title(f'{self.model_name}: Force run')
            self.params_exist = False
        else:
            self.params_exist_timedir, self.params_exist = search_dict_subdir(dict_data=self.params, basedir=self.basedir, pkl_name='params', value_dir_key=self.value_dir_key_before, both_dir_key=self.both_dir_key_before, after_subdir=pj(self.after_timedir, 'params'), ignore_key=self.ignore_key_list, pop=self.pop)
            if self.params_exist:
                print_title(f'{self.model_name}: Params exist')
                self.original_timedir = self.timedir
                self.timedir = pj(self.params_exist_timedir, self.after_timedir)
            else:
                print_title(f'{self.model_name}: Params do not exist')

    def check_simulation_results_exist(self):
        '''
        检查simulation结果是否存在

        注意:
        如果params_exist但是simulation_results不存在,说明之前的运行被中断了,需要重新运行,此时会新开一个timedir,而由于params_exist,timedir被设置为params_exist_timedir,所以需要在这里恢复timedir
        '''
        self.simulation_results_exist = self._check_all_results_saved('simulation')
        if self.simulation_results_exist:
            print_title(f'{self.model_name}: Simulation results exist')
        else:
            print_title(f'{self.model_name}: Simulation results do not exist')
            self.timedir = self.original_timedir # 恢复timedir

    def load_params(self):
        '''
        读取模型参数
        '''
        self.params = load_pkl(pj(self.params_dir, 'params'))
        self.set_params(self.params)


    def get_simulation_value(self, key=None, **kwargs):
        return self._get_results_value(results_prefix='simulation', key=key, **kwargs)


    def load_info_container(self):
        '''
        info container不一定存在,所以使用try except
        '''
        try:
            self.info_container = load_dict(pj(self.params_dir, 'info_container'))
            self.key_builder = self.info_container.key_builder
        except:
            pass
    # endregion

    # region save
    def save_code(self):
        '''
        备份模型代码
        '''
        # 保存code_file
        for code_file in self.code_file_list:
            save_code_copy(self.code_dir, code_file)
        # 保存CF_DIR下的所有函数
        for sub_file in get_subfile(CF_DIR):
            if sub_file.endswith('.py'):
                save_code_copy(self.code_dir, sub_file)
        print_title(f'{self.model_name}: Code saved')

    def save_params(self):
        '''
        保存模型参数
        '''
        if hasattr(self, 'params'):
            if check_all_file_exist_with_any_extension(pj(self.params_dir, 'params')):
                print_title(f'{self.model_name}: Params already saved')
            else:
                save_timed_dir_dict(dict_data=self.params, basedir=self.basedir, dict_name='params', value_dir_key=self.value_dir_key_before, both_dir_key=self.both_dir_key_before, after_timedir=pj(self.after_timedir, 'params'), current_time=self.current_time, pop=self.pop)
                print_title(f'{self.model_name}: Params saved')
        else:
            print_title(f'{self.model_name}: No params to save')

    def save_info_container(self):
        '''
        保存info_container
        '''
        if hasattr(self, 'info_container'):
            save_dict(self.info_container, pj(self.params_dir, 'info_container')) # info_container一般比较小但是包含的元素可能比较多,所以直接保存更好
            print_title(f'{self.model_name}: Info container saved')

    def save_simulation_results(self, filename='simulation_results', **kwargs):
        self._save_results(results_prefix='simulation', filename=filename, **kwargs)

def find_incomplete_model(dir_before_timedir, dir_after_timedir=None):
    '''
    寻找不完整的模型文件夹,以'simulation_results_saved'是否存在为判断标准
    '''
    incomplete_model_list = []

    sub_dir_list = get_subdir(dir_before_timedir)
    for sub_dir in sub_dir_list:
        mark_file = pj(sub_dir, dir_after_timedir, 'outcomes', 'simulation_results', 'simulation_results_saved')
        if check_all_file_exist_with_any_extension(mark_file):
            pass
        else:
            model_dir = pj(sub_dir, dir_after_timedir)
            print(f'Found incomplete model directory: {model_dir}')
            incomplete_model_list.append(model_dir)
    return incomplete_model_list

class MetaModelContainer(InstanceContainer):
    '''
    获取一个目录下的所有模型,功能如下:
    初始化时,自动略过不完整的模型
    统计参数
    '''
    def __init__(self, model_class, dir_before_timedir, dir_after_timedir=None):
        incomplete_model_list = find_incomplete_model(dir_before_timedir, dir_after_timedir)
        sub_dir_list = get_subdir(dir_before_timedir)
        instance_list = []
        for sub_dir in sub_dir_list:
            timedir = pj(sub_dir, dir_after_timedir)
            if timedir in incomplete_model_list:
                continue
            else:
                instance = model_class()
                instance.set_timedir(timedir)
                instance_list.append(instance)
        super().__init__(instance_list)

def get_iterable_ax(ax):
    '''
    不会改变原有输入的ax,返回一个可以迭代的ax
    '''
    if isinstance(ax, np.ndarray):
        return ax.flatten()
    elif isinstance(ax, list):
        return flatten_list(pure_list(ax))
    elif isinstance(ax, dict):
        return list(ax.values())
    else:
        return [ax]

def get_iterable_ax_for_decorator(ax):
    '''
    不会改变原有输入的ax,返回一个可以迭代的ax,但是当ax是单个ax时,返回本体(用于decorator时,可以保证输出不会因为这里搞了iterable而把单个ax的结果包装成list)
    '''
    if isinstance(ax, np.ndarray):
        return ax.flatten(), True
    elif isinstance(ax, list):
        return flatten_list(pure_list(ax)), True
    elif isinstance(ax, dict):
        return list(ax.values()), True
    else:
        return ax, False

def iterate_over_obj(func, obj_name):
    """
    装饰器：将 obj_name 参数转换为可迭代对象,并对每个元素调用原始函数.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取 obj_name 参数的值
        obj = get_param_value(func, args, kwargs, obj_name)
        iterable_obj, is_iterable = get_iterable_ax_for_decorator(obj)  # 将 obj 转换为可迭代对象
        if is_iterable:
            results = []

            # 获取函数签名,并动态绑定位置参数和关键字参数
            signature = inspect.signature(func)

            # 遍历 iterable_obj 的每个元素,并替换 obj_name 参数
            for item in iterable_obj:
                # 重新绑定参数,将 obj_name 替换为 item
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()  # 应用默认参数

                # 将 obj_name 参数替换为当前的 item
                bound_args.arguments[obj_name] = item

                # 调用原始函数,传入修改后的参数
                result = func(*bound_args.args, **bound_args.kwargs)
                results.append(result)

            # 重构结果,使其与原来的 obj 形状相匹配
            return rebuild_ax(results, obj)
        else:
            return func(*args, **kwargs)

    return wrapper

def iterate_over_axs(func):
    return iterate_over_obj(func, 'ax')

def iterate_over_subfigs(func):
    return iterate_over_obj(func, 'subfig')

def rebuild_ax(flatten_ax, original_ax):
    '''
    根据 original_ax 的结构,还原扁平化的 flatten_ax.
    '''
    if isinstance(original_ax, np.ndarray):
        # 将扁平化的 ax 重塑为原有 ndarray 的形状
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
        # 根据 list 的结构,递归还原
        return rebuild_list(flatten_ax, original_ax)
    elif isinstance(original_ax, dict):
        # 将 flatten_ax 的值与 dict 的 keys 重新组合
        return dict(zip(original_ax.keys(), flatten_ax))
    else:
        # 返回原有的 ax
        return flatten_ax

def squeeze_ax(ax):
    '''
    将array的ax压缩

    当然,也可以用于subfig的压缩
    '''
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
    """
    根据坐标轴范围过滤数据点

    参数:
    x -- x坐标值列表或数组
    y -- y坐标值列表或数组
    xlim -- 可选,x轴范围限制的元组 (x_min, x_max), 如果带有None则表示不限制该方向
    ylim -- 可选,y轴范围限制的元组 (y_min, y_max), 如果带有None则表示不限制该方向

    返回:
    x_filtered -- 过滤后的x坐标列表
    y_filtered -- 过滤后的y坐标列表
    """
    # 当任一限制条件存在时执行过滤
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

        filtered_data = [
            (xi, yi) for xi, yi in zip(x, y)
            if (xlim is None or (xlim[0] <= xi <= xlim[1])) and
               (ylim is None or (ylim[0] <= yi <= ylim[1]))
        ]
        # 处理过滤结果(包括空结果的情况)
        if filtered_data:
            x_filtered, y_filtered = zip(*filtered_data)
            return list(x_filtered), list(y_filtered)
        else:
            return [], []
    # 无限制条件时返回原始数据
    return x.copy(), y.copy()

def plt_scatter(ax, x, y, label=None, color=BLUE, vert=True, rasterized=False, rasterized_threshold=10000, xlim=None, ylim=None, linewidths=0., **kwargs):
    '''
    使用x和y绘制散点图,可以接受plt.scatter的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 散点图的颜色,默认为BLUE
    :param vert: 是否为垂直散点图,默认为True,即纵向
    :param rasterized: 是否对图像进行栅格化处理,默认为False(如果点数特别多,导致图像过大时,可以考虑开启)
    :param rasterized_threshold: 栅格化处理的阈值,默认为10000,当点数超过这个值时,会进行栅格化处理(如果不需要这个自动处理,可以设置为None或者False)
    :param xlim: x轴的范围,默认为None (用于仅仅绘画xlim范围内的点,而不是设定x轴范围)
    :param ylim: y轴的范围,默认为None
    :param linewidths: 线宽,默认为0. (防止点的s值小于linewidths时,出现空心点的情况)
    :param kwargs: 其他plt.scatter支持的参数

    注意
    -s是marker_size的平方
    -s是圆的面积,radius按照points的单位
    '''
    _print_invalid_value_warning('plt_scatter', x=x, y=y)
    x, y = filter_data_by_xylim(x, y, xlim=xlim, ylim=ylim)

    if not vert:
        x, y = y, x

    if (rasterized_threshold is not None) and (rasterized_threshold is not False):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            if len(x) > rasterized_threshold:
                rasterized = True

    # 画图
    if 'c' in kwargs:
        local_kwargs = kwargs.copy()
        local_kwargs['c'] = pure_list(local_kwargs['c'])
        if len(list_shape(local_kwargs['c'])) == 1:
            local_kwargs['c'] = [local_kwargs['c']]
        # 不输入color参数
        return ax.scatter(x, y, label=label, rasterized=rasterized, linewidths=linewidths, **local_kwargs)
    else:
        return ax.scatter(x, y, label=label, rasterized=rasterized, color=color, linewidths=linewidths, **kwargs)

def plt_line(ax, x, y, label=None, color=BLUE, vert=True, xlim=None, ylim=None, **kwargs):
    '''
    使用x和y绘制折线图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 折线图的颜色,默认为BLUE
    :param vert: 是否为垂直折线图,默认为True,即纵向
    :param xlim: x轴的范围,默认为None (用于仅仅绘画xlim范围内的点,而不是设定x轴范围)
    :param ylim: y轴的范围,默认为None
    :param kwargs: 其他plt.plot支持的参数
    '''
    _print_invalid_value_warning('plt_line', x=x, y=y)
    x, y = filter_data_by_xylim(x, y, xlim=xlim, ylim=ylim)

    # 画图
    if not vert:
        x, y = y, x
    return ax.plot(x, y, label=label, color=color, **kwargs)

def plt_bar(ax, x, y, label=None, color=BLUE, vert=True, equal_space=False, err=None, capsize=PLT_CAP_SIZE, ecolor=BLACK, elabel=None, width=BAR_WIDTH, **kwargs):
    '''
    使用x和y绘制柱状图,可以接受plt.bar的其他参数,此函数的特性是会根据x的值作为bar的位置,当x包含字符串或者equal_space=True时,会自动变成等距离排列.
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param color: 柱状图的颜色,默认为BLUE
    :param vert: 是否为垂直柱状图,默认为True,即纵向
    :param equal_space: 是否将x的值作为字符串处理,这将使得柱子等距排列,默认为False
    :param err: 误差线的数据,默认为None
    :param capsize: 误差线帽大小,默认为PLT_CAP_SIZE
    :param ecolor: 误差线颜色,默认为BLACK
    :param elabel: 误差线图例标签,默认为None
    :param width: 柱子宽度,默认为None(当vert=False时,此参数将自动赋值给height)
    :param kwargs: 其他plt.bar或plt.barh支持的参数
    '''
    _print_invalid_value_warning('plt_bar', x=x, y=y, err=err)
    if equal_space:
        # 将x的每个元素变为字符串
        x = [str(i) for i in x]

    error_kw = kwargs.pop('error_kw', None)
    if err is not None:
        if error_kw is None:
            error_kw = {}
        if 'elinewidth' not in error_kw:
            error_kw['elinewidth'] = ERRORBAR_LINEWIDTH
        if 'capthick' not in error_kw:
            error_kw['capthick'] = ERRORBAR_LINEWIDTH

        # 添加elabel
        add_errorbar(ax, x, y, err, vert=vert, label=elabel,
                     color=ecolor, capsize=capsize, **kwargs)

    if vert:
        # 绘制垂直柱状图
        return ax.bar(x, y, label=label, color=color, yerr=err, capsize=capsize, ecolor=ecolor, width=width, error_kw=error_kw, **kwargs)
    else:
        # 绘制水平柱状图
        return ax.barh(x, y, label=label, color=color, xerr=err, capsize=capsize, ecolor=ecolor, height=width, error_kw=error_kw, **kwargs)

def plt_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
    '''
    使用 get_hist 统计数据,再通过 plt_bar 绘制直方图
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据集
    :param bins: 直方图的箱数,默认为BIN_NUM
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param stat: 直方图的统计方法,默认为'probability'
    :param vert: 是否为垂直直方图,默认为True,即纵向
    :param kwargs: 其他plt.hist支持的参数
    '''
    plt_hist_instance = PltHist(ax, data, bins=bins, label=label, color=color, stat=stat, vert=vert, **kwargs)
    return plt_hist_instance.bar

class PltHist:
    '''
    用于绘制直方图的类(方便后续获取hist,edge,mid_point等)
    '''
    def __init__(self, ax, data, bins=BIN_NUM, label=None, color=BLUE, stat='probability', vert=True, **kwargs):
        '''
        使用 get_hist 统计数据,再通过 plt_bar 绘制直方图
        :param ax: matplotlib的轴对象,用于绘制图形
        :param data: 用于绘制直方图的数据集
        :param bins: 直方图的箱数,默认为BIN_NUM
        :param label: 图例标签,默认为None
        :param color: 直方图的颜色,默认为BLUE
        :param stat: 直方图的统计方法,默认为'probability'
        :param vert: 是否为垂直直方图,默认为True,即纵向
        :param kwargs: 其他plt.hist支持的参数
        '''
        _print_invalid_value_warning('plt_hist', data=data)
        self.hist, self.edge, self.mid_point = get_hist(data, bins, stat=stat)
        set_ax(ax, ylabel=stat)
        self.bar = plt_bar(ax, self.mid_point, self.hist, label=label, color=color, width=self.edge[1]-self.edge[0], vert=vert, **kwargs)

def plt_polygon(ax, xy, color=BLUE, fill=True, adjust_lim=True, **kwargs):
    '''
    在指定的轴上绘制多边形.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param xy: 多边形的顶点坐标.例子:xy = [(0, 0), (1, 1), (1, 0)]
    :param color: 多边形的颜色.
    :param fill: 是否填充多边形.(如果为False,则为多边形的边框,这时可以通过linewidth参数控制边框宽度)
    :param adjust_lim: 是否根据多边形的位置和大小调整轴的限制.
    :param kwargs: 其他matplotlib.Polygon支持的参数.
    ax.add_patch()方法绘制多边形,支持通过fill参数控制是否填充.
    '''
    _print_invalid_value_warning('plt_polygon', xy=xy)
    polygon = plt.Polygon(xy, color=color, fill=fill, **kwargs)
    if adjust_lim:
        for sub_xy in xy:
            ax.scatter(sub_xy[0], sub_xy[1], s=0, zorder=-1)
    return ax.add_patch(polygon)

def plt_bar_dict(ax, data, label=None, color=BLUE, width=BAR_WIDTH, vert=True, **kwargs):
    '''
    使用字典数据绘制柱状图
    '''
    x = range(len(data))
    y = list(data.values())
    set_ax(ax, xtick=x, xtick_label=list(data.keys()))
    return plt_bar(ax, x, y, label=label, color=color, width=width, vert=vert, **kwargs)

def sns_heatmap(ax, data, cmap=HEATMAP_CMAP, square=True, cbar=True, cbar_position=None, cbar_label=None, discrete_label=None, xtick_rotation=XTICK_ROTATION, ytick_rotation=YTICK_ROTATION, show_xtick=True, show_ytick=True, show_all_xtick=True, show_all_ytick=True, xtick_fontsize=None, ytick_fontsize=None, mask=None, mask_color=MASK_COLOR, mask_tick='mask', norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, heatmap_kwargs=None, cbar_kwargs=None):
    '''
    使用DataFrame或二维NumPy数组绘制热图.颜色条由add_side_colorbar单独绘制,
    而不是使用seaborn自带的颜色条.

    :param ax: matplotlib的轴对象.
    :param data: 二维pandas.DataFrame或numpy.ndarray.使用DataFrame时,index和columns会用作刻度标签.
    :param cmap: matplotlib Colormap.连续模式通常使用连续cmap;离散模式建议使用
        get_cmap(colors, continuous=False)创建ListedColormap.
    :param square: 是否令每个cell为正方形,默认为True.若希望热图填满ax的现有长宽比例,设为False.
    :param cbar: 是否通过add_side_colorbar在ax旁边添加颜色条.为True时返回colorbar列表,
        为False时不添加颜色条并返回None.
    :param cbar_position: 颜色条位置参数字典.位置可选'left', 'right', 'top', 'bottom'.
    :param cbar_label: 颜色条标题.
    :param discrete_label: 离散区间的文字标签.此参数只设置颜色条标签,不会令热图离散化.
        必须配合norm_mode='boundary'使用,标签数量应等于len(boundaries)-1.
    :param xtick_rotation: x轴刻度标签旋转角度.
    :param ytick_rotation: y轴刻度标签旋转角度.
    :param show_xtick: 是否显示x轴刻度标签.
    :param show_ytick: 是否显示y轴刻度标签.
    :param show_all_xtick: 是否用DataFrame的全部columns重设x轴刻度.
    :param show_all_ytick: 是否用DataFrame的全部index重设y轴刻度.
    :param xtick_fontsize: x轴刻度标签字号.
    :param ytick_fontsize: y轴刻度标签字号.
    :param mask: 与data形状一致的布尔矩阵.True表示该位置被遮盖.
    :param mask_color: mask区域的填充颜色.设为None时仍应用mask,但被遮盖的cell不绘制,
        因而露出ax背景;这不等同于禁用mask.
    :param mask_tick: 使用有色mask和颜色条时,mask附加颜色条的文字标签.
    :param norm_mode: 颜色归一化模式.可选'linear', 'log', 'symlog', 'two_slope',
        'boundary'.离散热图必须使用'boundary'.
    :param vmin: 显示范围下限.None时取应用mask后的可见数据最小值.
    :param vmax: 显示范围上限.None时取应用mask后的可见数据最大值.
    :param norm_kwargs: 传给get_norm的参数.常用参数为:symlog的linthresh,
        two_slope的vcenter,以及boundary的boundaries和ncolors.
    :param text_process: 颜色条文字处理配置.
    :param heatmap_kwargs: 传给sns.heatmap的其他参数,例如
        {'annot': True, 'fmt': '.2f', 'linewidths': 0.5}.不要在这里重复传入
        ax, cmap, square, cbar或norm,这些参数由本函数管理.
    :param cbar_kwargs: 传给add_side_colorbar的其他参数,例如
        {'display_edge_ticks': False}.不要重复传入本函数已经传递给颜色条的参数.

    连续模式示例::

        sns_heatmap(ax, data, norm_mode='linear', vmin=0, vmax=1)

    离散模式示例.区间为[0, 1), [1, 2), [2, 3],三个标签对应三个区间::

        colors = ['#d9f0ff', '#73b3d8', '#08306b']
        discrete_cmap = get_cmap(colors, continuous=False)
        sns_heatmap(
            ax,
            data,
            cmap=discrete_cmap,
            norm_mode='boundary',
            norm_kwargs={'boundaries': [0, 1, 2, 3], 'ncolors': 3},
            vmin=0,
            vmax=3,
            discrete_label=['low', 'medium', 'high'],
            cbar_kwargs={'display_edge_ticks': False},
        )

    推荐用法与约束:
    1. 连续热图:使用连续cmap,并根据数据选择linear、log、symlog或two_slope模式;
       不设置discrete_label.
    2. 离散热图:使用get_cmap(colors, continuous=False)创建离散cmap,同时使用
       boundary模式,并在norm_kwargs中提供boundaries和ncolors.
    3. 离散模式下,boundaries必须单调递增;ncolors应与离散cmap的颜色数一致,
       discrete_label数量必须等于len(boundaries)-1.建议将vmin和vmax分别设为
       boundaries的首值和末值.
    4. 不建议混用连续cmap与boundary模式,或离散cmap与非boundary模式;
       这些组合不保证热图区间、颜色和颜色条标签正确对应.

    seaborn热图默认令第0行显示在最上方.
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    cbar_kwargs = update_dict({}, cbar_kwargs)
    if xtick_fontsize is None:
        xtick_fontsize = TICK_SIZE
    if ytick_fontsize is None:
        ytick_fontsize = TICK_SIZE

    # 如果data是array,则转换为DataFrame
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
        # auto annot fontsize setup
        if 'annot' in local_heatmap_kwargs:
            annot_kws = dict(local_heatmap_kwargs.get('annot_kws') or {})
            if 'fontsize' not in annot_kws:
                ax_width, ax_height = get_ax_size(ax)
                from_height = suitable_tick_size(data.shape[0], ax_height)
                from_width = suitable_tick_size(data.shape[1], ax_width) / 3 # 考虑到宽度方向通常会更拥挤一些
                annot_kws['fontsize'] = min(from_height, from_width)
                local_heatmap_kwargs['annot_kws'] = annot_kws
    cbar_position = update_dict(CBAR_POSITION, cbar_position)

    # 获取norm
    norm = get_norm(vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs)

    # 绘制热图
    if np.allclose(vmin, vmax):
        local_cmap = get_cmap([cmap(0.5), cmap(0.5)])
        sns.heatmap(local_data, ax=ax, cmap=local_cmap,
                    square=square, cbar=False, norm=norm, **local_heatmap_kwargs)
    else:
        sns.heatmap(local_data, ax=ax, cmap=cmap,
                    square=square, cbar=False, norm=norm, **local_heatmap_kwargs)

    # 绘制mask矩阵
    if mask is not None and mask_color is not None:
        use_mask = True
        mask_cmap = get_cmap([mask_color, mask_color])
        sns.heatmap(mask, ax=ax, cmap=mask_cmap,
                    cbar=False, mask=~mask, square=square)
    else:
        use_mask = False

    # 设置x轴和y轴的刻度标签
    if show_xtick:
        if show_all_xtick:
            ax.set_xticks(np.arange(len(local_data.columns))+0.5)
            ax.set_xticklabels(
                local_data.columns, rotation=xtick_rotation, fontsize=xtick_fontsize)
        else:
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=xtick_rotation, fontsize=xtick_fontsize)
        ax.xaxis.set_tick_params(labelbottom=show_xtick)
    else:
        ax.set_xticks([])

    if show_ytick:
        if show_all_ytick:
            ax.set_yticks(np.arange(len(local_data.index))+0.5)
            ax.set_yticklabels(
                local_data.index, rotation=ytick_rotation, fontsize=ytick_fontsize)
        else:
            ax.set_yticklabels(ax.get_yticklabels(),
                               rotation=ytick_rotation, fontsize=ytick_fontsize)
        ax.yaxis.set_tick_params(labelleft=show_ytick)
    else:
        ax.set_yticks([])

    # 显示颜色条
    if cbar:
        cbars = add_side_colorbar(ax, mappable=None, cmap=cmap, cbar_position=cbar_position, cbar_label=cbar_label, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, discrete_label=discrete_label, use_mask=use_mask, mask_color=mask_color, mask_tick=mask_tick, **cbar_kwargs)
        return cbars

def add_errorbar(ax, x, y, err, label=None, color=BLACK, linestyle='None', capsize=PLT_CAP_SIZE, vert=True, equal_space=False, **kwargs):
    '''
    在指定位置添加误差线.当x包含字符串或者equal_space=True时,会自动变成等距离排列.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param x: 误差线的x坐标.也支持字符串,此时会自动转换为等距的.
    :param y: 误差线的y坐标.
    :param err: 误差线的数据.
    :param label: 误差线的标签,默认为None.
    :param color: 误差线的颜色,默认为BLACK.
    :param linestyle: 误差线的线型,默认为'None'.
    :param capsize: 误差线的线帽大小,默认为PLT_CAP_SIZE.capsize是相对于图的大小的,不会受到xlim和ylim的影响
    :param vert: 是否为垂直误差线,默认为True.
    :param equal_space: 是否将x的值作为字符串处理,这将使得柱子等距排列,默认为False
    :param kwargs: 传递给`ax.errorbar`的额外关键字参数.
    '''
    _print_invalid_value_warning('add_errorbar', x=x, y=y, err=err)

    if 'elinewidth' not in kwargs:
        kwargs['elinewidth'] = ERRORBAR_LINEWIDTH

    # 检查x中元素是否包含字符串
    if isinstance(x, (list, tuple, np.ndarray)):
        x = list(x)
        if any(isinstance(item, str) for item in x) or equal_space:
            # 如果x中包含字符串,则将其全部转换为字符串
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
    '''
    在指定位置添加一个新的inset ax;位置坐标是相对于原ax的坐标

    注意:
    如果使用 inset_mode='ax',则获得的 ax 和普通 ax 不同:
    - 内嵌 ax 会随原 ax 移动,不能单独移动.
    - fig.axes 不包含内嵌 ax.
    - 删除原 ax 时,内嵌 ax 也会随之不再显示.
    - 不论删除普通 ax 还是内嵌 ax,都应统一调用 rm_ax;rm_ax 会正确处理两种情况.
    - 内嵌 ax 无法作为 reparent_ax 的 parent_ax
      (这也可能是 reparent_ax 当前实现的限制).
    - 对原 ax 调用 set_visible(False) 时,内嵌 ax 也会被隐藏.
    '''
    valid_inset_modes = ('fig', 'ax')
    if inset_mode not in valid_inset_modes:
        raise ValueError(f"inset_mode must be one of {valid_inset_modes}, got {inset_mode!r}.")

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

        # 将new_ax添加到ax的custom_children中
        add_custom_child_to_ax(ax, new_ax, label)
        return new_ax

def add_ax(fig, left, right, bottom, top, label='add', **kwargs):
    '''
    在指定位置添加一个新的ax.
    :param fig: matplotlib的图形对象,用于绘制图形.
    :param left: 新ax的左边界位置.
    :param right: 新ax的右边界位置.
    :param bottom: 新ax的下边界位置.
    :param top: 新ax的上边界位置.
    :param kwargs: 传递给`fig.add_axes`的其他参数.比如sharex=some_ax, sharey=some_ax
    '''
    return fig.add_axes([left, bottom, right-left, top-bottom], label=label, **kwargs)

@iterate_over_axs
def add_side_ax(ax, position='right', relative_size=SIDE_PAD*2, pad=SIDE_PAD, sharex=None, sharey=None, hide_repeat_xaxis=False, hide_repeat_yaxis=False, label='side', inset_mode='fig', spine_mode='same', **kwargs):
    '''
    在指定位置添加一个新的ax,并可以选择共享x轴或y轴.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param position: 新ax的位置,可以是'right', 'left', 'top', 'bottom'.
    :param relative_size: 新ax的宽度或高度,相对于原ax的对应大小.
    :param pad: 新ax与原ax的间距,相对于原ax的对应大小.
    :param sharex: 与新ax共享x轴的ax对象.也可以是True,表示与原ax共享x轴.
    :param sharey: 与新ax共享y轴的ax对象.也可以是True,表示与原ax共享y轴.
    :param hide_repeat_xaxis: 如果共享x轴,是否隐藏重复的x轴标签,默认为True.
    :param hide_repeat_yaxis: 如果共享y轴,是否隐藏重复的y轴标签,默认为True.
    :param kwargs: 传递给
    '''
    valid_positions = ('right', 'left', 'top', 'bottom')
    if position not in valid_positions:
        raise ValueError(f"position must be one of {valid_positions}, got {position!r}.")

    if sharex is True:
        sharex = ax
    elif sharex is None or sharex is False:
        sharex = None
    if sharey is True:
        sharey = ax
    elif sharey is None or sharey is False:
        sharey = None

    new_pos = {'left': 0., 'right': 1., 'top': 1., 'bottom': 0.}
    if position == 'right':
        new_pos['left'] = 1. + pad
        new_pos['right'] = 1. + pad + relative_size
    elif position == 'left':
        new_pos['left'] = - relative_size - pad
        new_pos['right'] = - pad
    elif position == 'top':
        new_pos['top'] = 1. + pad + relative_size
        new_pos['bottom'] = 1. + pad
    elif position == 'bottom':
        new_pos['top'] = - pad
        new_pos['bottom'] = - relative_size - pad

    # 创建并返回新的ax,可能共享x轴或y轴
    new_ax = inset_ax(ax, left=new_pos['left'], right=new_pos['right'], bottom=new_pos['bottom'], top=new_pos['top'], sharex=sharex, sharey=sharey, label=label, inset_mode=inset_mode, **kwargs)

    # 设置新ax的spine
    if spine_mode == 'same':
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(ax.spines[location].get_visible())
    elif spine_mode == 'all':
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(True)
    elif spine_mode in ['none', 'hide'] or not spine_mode:
        for location in ['top', 'right', 'bottom', 'left']:
            new_ax.spines[location].set_visible(False)

    # 如果共享x轴且hide_repeat_xaxis为True,将特定的x轴标签设为不可见
    if sharex is not None and hide_repeat_xaxis:
        if position=='top':
            new_ax.xaxis.set_visible(False)
        elif position=='bottom':
            new_ax.xaxis.set_visible(False)

    # 如果共享y轴且hide_repeat_yaxis为True,将特定的y轴标签设为不可见
    if sharey is not None and hide_repeat_yaxis:
        if position=='right':
            new_ax.yaxis.set_visible(False)
        elif position=='left':
            new_ax.yaxis.set_visible(False)

    return new_ax

@iterate_over_axs
def rm_ax(ax):
    """删除指定的Axes对象"""
    ax.remove()

@iterate_over_axs
def set_ax_view_3d(ax, elev=ELEV, azim=AZIM):
    '''
    对单个或多个3D子图Axes应用统一的视角设置.

    参数:
    - ax: 单个Axes,或者np.ndarray,list,dict
    - elev: 视角的高度.
    - azim: 视角的方位角.
    '''
    if isinstance(ax, Axes3D):
        ax.view_init(elev=elev, azim=azim)

def get_ax_position_custom(ax):
    '''
    更加人性化,返回left, right, bottom, top
    '''
    left, bottom, width, height = ax.get_position().bounds
    right = left + width
    top = bottom + height
    return left, right, bottom, top

def set_ax_position_custom(ax, left, right, bottom, top):
    '''
    设置轴的位置.

    参数:
    - ax: matplotlib的Axes对象
    - left: 左边界的位置
    - right: 右边界的位置
    - bottom: 下边界的位置
    - top: 上边界的位置
    '''
    ax.set_position([left, bottom, right - left, top - bottom])

def set_relative_ax_position(ax, nrows=1, ncols=1, margin=None, squeeze=False):
    '''
    自动设置subplot的位置,使其在等分的图像中按照给定的比例占据空间.

    参数:
    - nrows: 子图的行数.
    - ncols: 子图的列数.
    - ax: 一个或一组matplotlib的Axes对象.
    - margin: 一个字典,定义了图像边缘的留白,包括left, right, bottom, top.
    - squeeze: ax是否被压缩,squeeze为True时,对于单个ax其就是axes对象;squeeze为False时,对于单个ax其是一个(1,1)的np.ndarray对象
    '''
    if margin is None:
        margin = MARGIN.copy()

    # 计算每个子图的宽度和高度(相对于整个图像的宽度和高度,所以是比例值)
    subplot_width = 1 / ncols
    subplot_height = 1 / nrows
    ax_width = (margin['right'] - margin['left']) * subplot_width
    ax_height = (margin['top'] - margin['bottom']) * subplot_height

    if nrows > 1 and ncols > 1:
        # 对于每个子图,计算其位置并设置
        for row in range(nrows):
            for col in range(ncols):
                left = margin['left'] / ncols + col * subplot_width
                bottom = margin['bottom'] / nrows + \
                    (nrows - row - 1) * subplot_height

                # 设置子图的位置
                ax[row, col].set_position(
                    [left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols > 1:
        for col in range(ncols):
            left = margin['left'] / ncols + col * subplot_width
            bottom = margin['bottom']

            # 设置子图的位置
            if squeeze:
                ax[col].set_position([left, bottom, ax_width, ax_height])
            else:
                ax[0, col].set_position([left, bottom, ax_width, ax_height])
    if nrows > 1 and ncols == 1:
        for row in range(nrows):
            left = margin['left']
            bottom = margin['bottom'] / nrows + (nrows - row - 1) * subplot_height

            # 设置子图的位置
            if squeeze:
                ax[row].set_position([left, bottom, ax_width, ax_height])
            else:
                ax[row, 0].set_position([left, bottom, ax_width, ax_height])
    if nrows == 1 and ncols == 1:
        left = margin['left']
        bottom = margin['bottom']

        # 设置子图的位置
        if squeeze:
            ax.set_position([left, bottom, ax_width, ax_height])
        else:
            ax[0, 0].set_position([left, bottom, ax_width, ax_height])

    return ax

@iterate_over_axs
def align_ax(ax, ref_ax, align_mode='horizontal', keep_size=False):
    '''
    将axs中ax的position对齐到ref_ax的position
    align_mode:
        'horizontal', 'vertical' - 改变高度和宽度,horizontal则左右对齐(ax的上下框对齐),vertical则上下对齐(ax的左右框对齐)
        'left', 'right', 'top', 'bottom' - 不改变高度和宽度,只改变位置去对齐到需要的位置
        'all' - 改变高度和宽度,同时改变位置去对齐到需要的位置(即直接对齐到ref_ax的position)
    keep_size: 是否保持ax的大小不变,默认为False,即改变大小(其他点固定,拉伸至指定点);假如为True,则保持大小,只改变位置(平移)
    '''
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
    '''
    向某个方向移动ax的位置

    参数:
    -keep_size: 是否保持ax的大小不变,默认为False,即改变大小(其他点固定,拉伸至指定点);假如为True,则保持大小,只改变位置(平移)
    '''
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

        left, right, bottom, top = position_dict['left'], position_dict['right'], position_dict['bottom'], position_dict['top']
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
    '''
    根据给定的边界和比例计算GridSpec的坐标
    '''
    # 设定默认的比例和间距
    if width_ratios is None:
        width_ratios = [1] * ncols
    if height_ratios is None:
        height_ratios = [1] * nrows
    if wspace is None:
        wspace = plt.rcParams['figure.subplot.wspace']
    if hspace is None:
        hspace = plt.rcParams['figure.subplot.hspace']


    # Calculate total space in each direction
    width_total = right - left
    height_total = top - bottom

    # Calculate sum of ratios and effective space for grids
    width_ratios_sum = sum(width_ratios)
    height_ratios_sum = sum(height_ratios)

    # Calculate grid widths and heights without spacing
    cell_widths = [(r / width_ratios_sum) * (width_total - (len(width_ratios) - 1) * wspace) for r in width_ratios]
    cell_heights = [(r / height_ratios_sum) * (height_total - (len(height_ratios) - 1) * hspace) for r in height_ratios]

    # Initialize lists to store cell boundaries
    x_coords = [left]
    y_coords = [top]  # Starting from top for a top-down layout

    # Calculate x coordinates for each column
    for width in cell_widths:
        x_coords.append(x_coords[-1] + width + wspace)

    # Calculate y coordinates for each row
    for height in cell_heights:
        y_coords.append(y_coords[-1] - height - hspace)

    # Collect coordinates for each grid cell
    # coordinates = []
    # for i in range(len(height_ratios)):
    #     row = []
    #     for j in range(len(width_ratios)):
    #         cell_left = x_coords[j]
    #         cell_right = x_coords[j + 1] - wspace
    #         cell_bottom = y_coords[i + 1] + hspace
    #         cell_top = y_coords[i]
    #         row.append((cell_left, cell_right, cell_bottom, cell_top))
    #     coordinates.append(row)
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
    '''
    在ax的位置为基础获取一个GridSpec对象,然后根据GridSpec对象获取所有的ax对象.可以用于切分ax.

    参数:
    -keep_original: 是否保留原始的ax,默认为False,即不保留(将会被rm);如果为True,则会保留原始的ax;如果为index,则会将原先的ax放在返回的ax中的index位置
    '''
    if inset_mode == 'fig':
        gs = get_gs_inside_ax(ax, nrows=nrows, ncols=ncols, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
        label = cat(label, ax.get_label())
        sub_ax = get_all_ax_from_gs(gs, squeeze=squeeze, label=label, **kwargs)
        if keep_original is True:
            # 不做处理
            pass
        elif keep_original is False:
            rm_ax(ax)
            sub_ax = share_axis(sub_ax, sharex=sharex, sharey=sharey)
        else:
            # 重新share_axis到original_ax(此时不支持row,col)
            if sharex in ['row', 'col'] or sharey in ['row', 'col']:
                print('sharex or sharey can not be row or col in keep_original mode')
            sub_ax = share_axis_to_target(sub_ax, ax, sharex=sharex, sharey=sharey)
            # 将orginal_ax在图中放到index的ax所处的位置
            align_ax(ax, sub_ax[keep_original], align_mode='all')
            # 将sub_ax[keep_original]移除
            rm_ax(sub_ax[keep_original])
            # 将original_ax替换到sub_ax中
            sub_ax[keep_original] = ax
    elif inset_mode == 'ax':
        coords = calculate_gs_coordinates(ncols=ncols, nrows=nrows, left=0., right=1., bottom=0., top=1., wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)
        # 根据coords创建ax
        sub_ax = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                local_label = cat(label, f'row{i}_col{j}')
                sub_ax[i, j] = inset_ax(ax, coords[i][j][0], coords[i][j][1], coords[i][j][2], coords[i][j][3], label=local_label, inset_mode='ax', **kwargs)
        if keep_original is True:
            # 不做处理
            pass
        elif keep_original is False:
            # 注意,这里不可以把原ax直接删除,因为这会导致sub_ax中的ax也被删除,只能尽量把原ax的属性删除,就设计属性而言,用户不应该在原ax上画任何图像(因为原ax是要被分割的,不是用来画图的)
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
    '''
    添加双边缘轴
    '''
    if x_side_ax_position == 'top':
        height_ratios = [x_side_ax_size, 1-x_side_ax_size]
        x_side_index = 0 # 取第0行
    elif x_side_ax_position == 'bottom':
        height_ratios = [1-x_side_ax_size, x_side_ax_size]
        x_side_index = 1

    if y_side_ax_position == 'right':
        width_ratios = [1-y_side_ax_size, y_side_ax_size]
        y_side_index = 1
    elif y_side_ax_position == 'left':
        width_ratios = [y_side_ax_size, 1-y_side_ax_size]
        y_side_index = 0

    # 目前ax模式不支持keep_original为一个特定索引(后续可能会支持)
    if inset_mode == 'fig':
        keep_original = (1-x_side_index, 1-y_side_index)
        sub_ax = split_ax_by_gs(ax, nrows=2, ncols=2, hspace=x_side_ax_pad, wspace=y_side_ax_pad, height_ratios=height_ratios, width_ratios=width_ratios, keep_original=keep_original, label=label, inset_mode=inset_mode)
    elif inset_mode == 'ax':
        print_title('inset_mode=ax need to delete original ax and re-create an ax in the needed position, thus not recommended, please insure that other process on ax is done after this function')
        sub_ax = split_ax_by_gs(ax, nrows=2, ncols=2, hspace=x_side_ax_pad, wspace=y_side_ax_pad, height_ratios=height_ratios, width_ratios=width_ratios, keep_original=False, label=label, inset_mode=inset_mode)

    x_side_ax = sub_ax[x_side_index, 1-y_side_index]
    y_side_ax = sub_ax[1-x_side_index, y_side_index]
    ax = sub_ax[1-x_side_index, 1-y_side_index]

    x_side_ax = share_axis_to_target(x_side_ax, ax, sharex=True, sharey=False)
    y_side_ax = share_axis_to_target(y_side_ax, ax, sharex=False, sharey=True)

    rm_ax(sub_ax[x_side_index, y_side_index])
    return x_side_ax, y_side_ax, ax

def merge_ax(axs, rm_mode='rm_axis', label='merge'):
    '''
        合并给定的轴对象列表或数组为一个轴对象.

        :param axs: 要合并的轴对象列表或数组
        :param rm_mode: 是否删除原始的轴对象.默认为'rm_axis',即删除原始的轴对象;假如为'rm_ax',则删除整个ax;如果是其他值,则不删除原始的轴对象
    '''
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
    '''
    向ax中添加一个自定义的子元素(放置于ax.custom_children中)
    '''
    if not hasattr(ax, 'custom_children'):
        ax.custom_children = {}
    if child_name not in ax.custom_children:
        ax.custom_children[child_name] = child
    else:
        new_child_name = child_name + '_new'
        add_custom_child_to_ax(ax, child, new_child_name)

def get_extreme_ax_position(axs, position):
    '''
    获取多个ax的位置的极值.
    :param axs: 多个ax对象.
    :param position: 位置参数,可选'left', 'right', 'top', 'bottom'.
    '''
    axs = get_iterable_ax(axs)
    if position not in ['left', 'right', 'top', 'bottom']:
        raise ValueError('position参数错误')
    if position == 'left':
        return min([ax.get_position().x0 for ax in axs])
    elif position == 'right':
        return max([ax.get_position().x1 for ax in axs])
    elif position == 'top':
        return max([ax.get_position().y1 for ax in axs])
    elif position == 'bottom':
        return min([ax.get_position().y0 for ax in axs])

def get_linear_but_log_formatter(base='10', label_format='auto'):
    '''
    假如散点等元素是log后作画的,ax本身的scale是linear,则使用这个formatter可以把tick显示为log的形式.
    base: 指数的底数,默认是10.
    label_format: 控制指数部分的格式,例如 '%.0f' 表示整数形式,'%.2f' 表示保留两位小数.默认是'auto',即自动选择.
    '''
    def linear_but_log_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(x)
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return r"${}^{{{}}}$".format(base, local_label_format % x)

    return FuncFormatter(linear_but_log_formatter)

@iterate_over_axs
def set_linear_but_log_axis(ax, axis=None, base='10', label_format='auto'):
    '''
    注意:最好放在set ax后,不然有可能小数点的估计会错误
    '''
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
    '''
    假设ax本身的scale是log的并且base为e,则使用这个formatter可以把tick显示为e的形式(否则会有很多小数点)
    label_format: 控制指数部分的格式,例如 '%.0f' 表示整数形式,'%.2f' 表示保留两位小数.默认是'auto',即自动选择.
    '''
    def log_e_formatter(x, pos):
        if label_format == 'auto':
            decimal_num = get_decimal_num(np.log(x))
            local_label_format = '%.' + str(decimal_num) + 'f'
        else:
            local_label_format = label_format
        return r"$e^{{{}}}$".format(local_label_format % np.log(x))

    return FuncFormatter(log_e_formatter)

@iterate_over_axs
def set_log_e_axis(ax, axis=None, label_format='auto'):
    '''
    注意:最好放在set ax后,不然有可能小数点的估计会错误
    '''
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
    '''
    获取对称正数的formatter
    '''
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
    '''
    将轴设置为对称正数
    '''
    if axis is None:
        axis = ['x', 'y']
    if isinstance(axis, str):
        axis = [axis]
    for axi in axis:
        if axi == 'x':
            # 将x轴设置为对称
            try:
                if bound is None:
                    bound = max(abs(ax.get_xlim()))
                ax.set_xlim(-bound, bound)
            except:
                pass

            # 获取label_format
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_xticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format

            ax.xaxis.set_major_formatter(get_sym_positive_formatter(label_format=local_label_format))
        if axi == 'y':
            # 将y轴设置为对称
            try:
                if bound is None:
                    bound = max(abs(ax.get_ylim()))
                ax.set_ylim(-bound, bound)
            except:
                pass

            # 获取label_format
            if label_format == 'auto':
                max_decimal_num = get_max_decimal_num(ax.get_yticks())
                local_label_format = '%.' + str(max_decimal_num) + 'f'
            else:
                local_label_format = label_format

            ax.yaxis.set_major_formatter(get_sym_positive_formatter(label_format=local_label_format))

def get_norm(norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None):
    '''
    根据给定的模式创建一个matplotlib颜色规范化对象.

    参数:
    - norm_mode (str): 规范化模式.可以是'linear','log','symlog','two_slope','boundary'.
    - vmin (float or None, optional): 规范化范围的最小值.
    - vmax (float or None, optional): 规范化范围的最大值.
    - norm_kwargs (dict or None, optional): 规范化函数的其他关键字参数.

    返回:
    - norm: 基于指定模式的matplotlib规范化对象.
    '''
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
                vmin = 1e-1
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
    '''
    生成颜色映射.
    :param colors: 颜色列表.
    :param continuous: 是否为连续颜色映射,默认为True.
    '''
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
    '''
    在指定ax添加颜色条.目前设置norm_mode为'boundary'时,最好输入一个离散的cmap,否则在log模式下会出现问题.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param mappable: 用于绘制颜色条的对象,默认为None.
    :param cmap: 颜色条的颜色映射,默认为CMAP.可以是离散的或连续的.
    :param ticks: 颜色条的刻度,默认为None,自动设置(当不想显示tick的时候可以输入[]);如果输入了,则会使用输入的刻度
    :param tick_labels: 颜色条的刻度标签,默认为None,自动设置;如果输入了,则会使用输入的刻度标签
    :param discrete_label: 离散颜色条的标签,默认为None.
    :param display_edge_ticks: 是否显示颜色条的边缘刻度,默认为True.(只在norm_mode为'boundary'时并且discrete_label不为None时有效)
    :param cbar_position: 颜色条的位置,默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param cbar_label: 颜色条的标签,默认为None.
    :param use_mask: 是否使用mask颜色条,默认为False.
    :param mask_color: mask颜色条的颜色,默认为MASK_COLOR.
    :param mask_pad: mask_colorbar和colorbar的间距,默认为0.
    :param mask_cbar_ratio: mask颜色条的比例,默认为None.对于连续会自动设置为0.2,对于离散会自动设置为1/(discrete+1)
    :param mask_tick: mask颜色条的标签,默认为'mask'.
    :param mask_tick_loc: mask颜色条的标签位置,默认为None.
    :param label_size: 颜色条标签的字体大小,默认为CBAR_LABEL_SIZE.(这个是指colorbar旁边的colorbar标签的大小)
    :param tick_size: 颜色条刻度标签的字体大小,默认为CBAR_TICK_SIZE.
    :param adjust_tick_size: 是否根据颜色条的数量自动调整刻度标签的大小,默认为True.
    :param tick_proportion: 调整时的比例,默认为TICK_PROPORTION.
    :param label_kwargs: 颜色条标签的其他参数,默认为None.
    :param norm_mode: 颜色条的归一化模式,默认为'linear'.可选'linear', 'log', 'symlog', 'twoslope'.
    :param vmin: 颜色条的最小值,默认为None.
    :param vmax: 颜色条的最大值,默认为None.
    :param norm_kwargs: 归一化的其他参数,默认为None.具体会根据不同的norm_mode在内部设置默认值.
    :param text_process: 文本处理函数,默认为TEXT_PROCESS.
    :param round_digits: 刻度标签的小数位数,默认为ROUND_DIGITS.
    :param add_leq: 是否在最小值处添加'<=',默认为False.
    :param add_geq: 是否在最大值处添加'>=',默认为False.
    :param inset_mode: 插入模式,默认为'fig'.可选'fig', 'ax'.(指的是如果分割ax获得mask_ax和cbar_ax时,是插入到fig中还是ax中)
    '''
    # 设定默认参数
    norm_kwargs = update_dict({}, norm_kwargs)
    if mappable is None:
        # 根据norm_mode设置norm
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

    # 更新默认值
    cbar_position = update_dict(CBAR_POSITION, cbar_position)
    text_process = update_dict(TEXT_PROCESS, text_process)

    # 根据vmin和vmax来clip这个mappable的范围
    mappable.set_clim(vmin, vmax)

    # 根据vmin和vmax判断是否add_leq和add_geq
    if vmin < mappable.norm.vmin:
        add_leq = True
    if vmax > mappable.norm.vmax:
        add_geq = True

    # 设置mask_cbar_ratio
    if mask_cbar_ratio is None:
        mask_cbar_ratio = 0.2
        if norm_mode == 'boundary':
            mask_cbar_ratio = 1/mappable.norm.boundaries.size

    # 格式化cbar_label和mask_tick
    cbar_label = format_text(cbar_label, text_process=text_process)
    mask_tick = format_text(mask_tick, text_process=text_process)

    # 格式化discrete_label
    if discrete_label is not None:
        discrete_label = [format_text(label, text_process=text_process) for label in discrete_label]

    # 获得orientation
    if cbar_position['position'] in ['right', 'left']:
        orientation = 'vertical'
    if cbar_position['position'] in ['top', 'bottom']:
        orientation = 'horizontal'

    # 获取cbar_ax和mask_ax
    if use_mask:
        if cbar_position['position'] in ['right', 'left']:
            if mask_pos == 'start':
                cbar_ax, mask_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=mask_pad, height_ratios=[1-mask_cbar_ratio, mask_cbar_ratio], label='cbar_mask', inset_mode=inset_mode)
            if mask_pos == 'end':
                mask_ax, cbar_ax = split_ax_by_gs(ax, nrows=2, ncols=1, hspace=mask_pad, height_ratios=[mask_cbar_ratio, 1-mask_cbar_ratio], label='mask_cbar', inset_mode=inset_mode)
        if cbar_position['position'] in ['top', 'bottom']:
            if mask_pos == 'start':
                mask_ax, cbar_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=mask_pad, width_ratios=[mask_cbar_ratio, 1-mask_cbar_ratio], label='mask_cbar', inset_mode=inset_mode)
            if mask_pos == 'end':
                cbar_ax, mask_ax = split_ax_by_gs(ax, nrows=1, ncols=2, wspace=mask_pad, width_ratios=[1-mask_cbar_ratio, mask_cbar_ratio], label='cbar_mask', inset_mode=inset_mode)

        # 去掉ax的坐标轴
        ax.axis('off')

        # 创建一个ScalarMappable对象,使用一个简单的颜色映射和Normalize对象,因为我们要显示一个纯色的colorbar,所以颜色映射范围不重要,只需确保vmin和vmax不同即可
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap=get_cmap([mask_color, mask_color]))

        # 创建colorbar,将ScalarMappable作为输入
        mask_cbar = plt.colorbar(sm, cax=mask_ax, orientation=orientation)

        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation=orientation)
    else:
        cbar = plt.colorbar(mappable, cax=ax, orientation=orientation)

    # 设置mask的刻度位置
    if use_mask:
        if norm_mode == 'boundary' and discrete_label is None:
            mask_cbar.set_ticks([0.0])   # 设置刻度位置在最侧边
        else:
            mask_cbar.set_ticks([0.5])   # 设置刻度位置在正中间

        # 假如输入了mask_tick_loc,则设置mask_cbar的刻度位置
        if mask_tick_loc is not None:
            mask_cbar.set_ticks([mask_tick_loc])

    # 设置colorbar的标签
    cbar.set_label(cbar_label, fontsize=label_size, **label_kwargs)

    # 在norm_mode 'boundary' 时,用户需要输入norm_kwargs
    if norm_mode == 'boundary':
        if 'boundaries' not in norm_kwargs:
            raise ValueError('当norm_mode为boundary时,需要在norm_kwargs中输入boundaries参数')
        if 'ncolors' not in norm_kwargs:
            raise ValueError('当norm_mode为boundary时,需要在norm_kwargs中输入ncolors参数')

    # 假如颜色映射是一个离散的,则根据boundaries来设置ticks(假如输入了discrete_label,则插入中间刻度)
    if norm_mode == 'boundary':
        if ticks is None:
            ticks = mappable.norm.boundaries
            if discrete_label is not None:
                ticks = insert_mid(ticks)
                if not display_edge_ticks:
                    # 去掉边缘的刻度,只保留中间的刻度
                    ticks = ticks[1::2]
    else:
        if ticks is None:
            ticks = cbar.get_ticks()
        # 得到ticks中位于vmin和vmax之间的刻度
        ticks = [tick for tick in ticks if vmin <= tick <= vmax]

        # 当add_leq和add_geq为True时,强行添加末端的刻度
        if add_leq:
            if not np.allclose(vmin, ticks[0]):
                ticks = [vmin] + ticks
        if add_geq:
            if not np.allclose(vmax, ticks[-1]):
                ticks = ticks + [vmax]

    # 设置tick
    cbar.set_ticks(ticks)

    # format cbar的刻度标签
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

    # 根据discrete_label进一步调整tick_labels
    if norm_mode == 'boundary':
        if discrete_label is not None:
            if display_edge_ticks:
                # 将tick_labels中间的刻度替换为discrete_label
                for i in range(len(ticks)-1):
                    if i % 2 == 1:
                        tick_labels[i] = discrete_label[i//2]
            else:
                # 将tick_labels全部替换为discrete_label
                tick_labels = discrete_label.copy()

    # 调整tick_size
    if adjust_tick_size:
        num_ticks = len(cbar.get_ticks())
        width, height = get_ax_size(cbar.ax)
        if cbar_position['position'] in ['right', 'left']:
            plt_size = height
        elif cbar_position['position'] in ['top', 'bottom']:
            plt_size = width
            tick_size = suitable_tick_size(num_ticks, plt_size, tick_size, tick_proportion)

    # 设置<=和>=
    if add_leq:
        if np.allclose(vmin, ticks[0]):
            tick_labels[0] = '≤' + tick_labels[0]
    if add_geq:
        if np.allclose(vmax, ticks[-1]):
            tick_labels[-1] = '≥' + tick_labels[-1]

    # 设置cbar刻度标签,并调节字体大小
    cbar.set_ticklabels(tick_labels, fontsize=tick_size)

    # 设置mask刻度标签,并调节字体大小
    if use_mask:
        mask_cbar.set_ticklabels([mask_tick], fontsize=tick_size)

    # 调整tick的方向
    if use_mask:
        cbar_list = [cbar, mask_cbar]
    else:
        cbar_list = [cbar]
    for cb in cbar_list:
        if cbar_position['position']=='right':
            pass
        elif cbar_position['position']=='left':
            cb.ax.yaxis.set_ticks_position('left')
        elif cbar_position['position']=='top':
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.set_xticks(cb.ax.get_xticks())
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        elif cbar_position['position']=='bottom':
            cb.ax.xaxis.set_ticks_position('bottom')
            cb.ax.set_xticks(cb.ax.get_xticks())
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=90)
        # 将spine全部去掉
        cb.outline.set_visible(False)

        # 将minor ticks全部去掉
        cb.ax.xaxis.set_tick_params(which='minor', length=0)
        cb.ax.yaxis.set_tick_params(which='minor', length=0)

    # return cbar
    if use_mask:
        add_custom_child_to_ax(ax, mask_cbar, f'mask_cbar_{mask_tick}')
        add_custom_child_to_ax(ax, cbar, f'cbar_{cbar_label}')
        return [cbar, mask_cbar]
    else:
        add_custom_child_to_ax(ax, cbar, f'cbar_{cbar_label}')
        return [cbar]

@iterate_over_axs
def add_side_colorbar(ax, mappable=None, cmap=CMAP, ticks=None, tick_labels=None, discrete_label=None, display_edge_ticks=True, cbar_position=None, cbar_label=None, use_mask=False, mask_color=MASK_COLOR, mask_pos='start', mask_pad=0, mask_cbar_ratio=None, mask_tick='mask', mask_tick_loc=None, label_size=None, tick_size=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, label_kwargs=None, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, text_process=None, formatter=None, formatter_kwargs=None, round_digits=ROUND_DIGITS, round_format_type=ROUND_FORMAT, add_leq=False, add_geq=False, inset_mode='fig'):
    '''
    在指定ax的旁边添加颜色条.特别注意,对于离散的cmap,用户一定要提供对应的discrete_num
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param mappable: 用于绘制颜色条的对象,默认为None.
    :param cmap: 颜色条的颜色映射,默认为CMAP.可以是离散的或连续的,须与discrete参数相符合.
    :param continuous_tick_num: 连续颜色条的刻度数量,默认为None,不作设置.
    :param discrete: 是否为离散颜色条,默认为False.
    :param discrete_num: 离散颜色条的数量,默认为5.
    :param discrete_label: 离散颜色条的标签,默认为None.
    :param display_edge_ticks: 是否显示颜色条的边缘刻度,默认为True.(只在离散颜色条下有效)
    :param display_center_ticks: 是否显示颜色条的中心刻度,默认为False.(只在离散颜色条下有效,并且假如discrete_label被设置了,即使display_center_ticks为True,也不会显示中心刻度)
    :param cbar_position: 颜色条的位置,默认为None,即使用默认位置;position参数可选'left', 'right', 'top', 'bottom'
    :param cbar_label: 颜色条的标签,默认为None.
    :param use_mask: 是否使用mask颜色条,默认为False.
    :param mask_color: mask颜色条的颜色,默认为MASK_COLOR.
    :param mask_pad: mask_colorbar和colorbar的间距,默认为0.
    :param mask_cbar_ratio: mask颜色条的比例,默认为None.对于连续会自动设置为0.2,对于离散会自动设置为1/(discrete+1)
    :param mask_tick: mask颜色条的标签,默认为'mask'.
    :param mask_tick_loc: mask颜色条的标签位置,默认为None.
    :param label_size: 颜色条标签的字体大小,默认为CBAR_LABEL_SIZE.
    :param tick_size: 颜色条刻度标签的字体大小,默认为CBAR_TICK_SIZE.
    :param adjust_tick_size: 是否根据颜色条的数量自动调整刻度标签的大小,默认为True.
    :param tick_proportion: 调整时的比例,默认为TICK_PROPORTION.
    :param label_kwargs: 颜色条标签的其他参数,默认为None.
    :param norm_mode: 颜色条的归一化模式,默认为'linear'.可选'linear', 'log', 'symlog', 'twoslope'.
    :param vmin: 颜色条的最小值,默认为None.
    :param vmax: 颜色条的最大值,默认为None.
    :param norm_kwargs: 归一化的其他参数,默认为None.具体会根据不同的norm_mode在内部设置默认值.
    :param text_process: 文本处理函数,默认为TEXT_PROCESS.
    :param round_digits: 刻度标签的小数位数,默认为ROUND_DIGITS.
    :param add_leq: 是否在最小值处添加'<=',默认为False.
    :param add_geq: 是否在最大值处添加'>=',默认为False.
    '''
    # 更新默认值
    if isinstance(ax, Axes3D):
        cbar_position = update_dict(CBAR_POSITION_3D, cbar_position)
    else:
        cbar_position = update_dict(CBAR_POSITION, cbar_position)

    side_ax = add_side_ax(ax, cbar_position['position'], cbar_position['size'], cbar_position['pad'], inset_mode=inset_mode)
    return add_colorbar(side_ax, mappable=mappable, cmap=cmap, ticks=ticks, tick_labels=tick_labels, discrete_label=discrete_label, display_edge_ticks=display_edge_ticks, cbar_position=cbar_position, cbar_label=cbar_label, use_mask=use_mask, mask_color=mask_color, mask_pos=mask_pos, mask_pad=mask_pad, mask_cbar_ratio=mask_cbar_ratio, mask_tick=mask_tick, mask_tick_loc=mask_tick_loc, label_size=label_size, tick_size=tick_size, adjust_tick_size=adjust_tick_size, tick_proportion=tick_proportion, label_kwargs=label_kwargs, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, text_process=text_process, formatter=formatter, formatter_kwargs=formatter_kwargs, round_digits=round_digits, round_format_type=round_format_type, add_leq=add_leq, add_geq=add_geq, inset_mode=inset_mode)

def add_vline(ax, x, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=None, **kwargs):
    '''
    在指定位置添加垂直线.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param x: 垂直线的x坐标.
    :param label: 垂直线的标签,默认为None.
    :param color: 垂直线的颜色,默认为RED.
    :param linestyle: 垂直线的线型,默认为AUXILIARY_LINE_STYLE.
    :param linewidth: 垂直线的线宽,默认为LINE_WIDTH.
    :param kwargs: 传递给`ax.axvline`的额外关键字参数.
    '''
    _print_invalid_value_warning('add_vline', x=x)
    if linewidth is None:
        linewidth = LINE_WIDTH

    # 画图
    return ax.axvline(x, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

def add_hline(ax, y, label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=None, **kwargs):
    '''
    在指定位置添加水平线.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param y: 水平线的y坐标.
    :param label: 水平线的标签,默认为None.
    :param color: 水平线的颜色,默认为RED.
    :param linestyle: 水平线的线型,默认为AUXILIARY_LINE_STYLE.
    :param linewidth: 水平线的线宽,默认为LINE_WIDTH.
    :param kwargs: 传递给`ax.axhline`的额外关键字参数.
    '''
    _print_invalid_value_warning('add_hline', y=y)
    if linewidth is None:
        linewidth = LINE_WIDTH

    # 画图
    return ax.axhline(y, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

def add_sline(ax, slope, intercept, reference_point='auto', label=None, color=RED, linestyle=AUXILIARY_LINE_STYLE, linewidth=None, **kwargs):
    '''
    按照斜率和截距添加直线.(贯穿整个ax)

    add_sline意为add_slope_line
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param slope: 直线的斜率.
    :param intercept: 直线的截距.
    :param reference_point: 直线的参考点,默认为'auto'.可输入浮点数
    :param label: 直线的标签,默认为None.
    :param color: 直线的颜色,默认为RED.
    :param linestyle: 直线的线型,默认为AUXILIARY_LINE_STYLE.
    :param linewidth: 直线的线宽,默认为LINE_WIDTH.
    :param kwargs: 传递给`ax.axline`的额外关键字参数.
    '''
    _print_invalid_value_warning('add_sline', slope=slope, intercept=intercept, reference_point=reference_point)
    if linewidth is None:
        linewidth = LINE_WIDTH

    if reference_point == 'auto':
        reference_point = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    reference_intercept = slope * reference_point + intercept
    return ax.axline((reference_point, reference_intercept), slope=slope, label=label, color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

def add_patch(ax, patch, auto_scale=True):
    '''
    在ax上添加patch.

    参数:
    - ax: matplotlib的Axes对象,用于绘制图形.
    - patch: matplotlib的Patch对象,用于添加到ax.
    - auto_scale: 是否自动调整坐标轴范围,默认为True.(如果为False,xlim和ylim不会被影响)
    '''
    ax.add_patch(patch)
    if auto_scale:
        ax.autoscale_view()

def add_circle_patch(ax, center, radius, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=None, **kwargs):
    '''
    创建圆形patch.

    参数:
    - center: 圆心坐标.
    - radius: 圆半径.
    - facecolor: 填充颜色,默认为'none'
    - edgecolor: 边框颜色,默认为BLACK
    - auto_scale: 是否自动调整坐标轴范围,默认为True
    '''
    _print_invalid_value_warning('add_circle_patch', center=center, radius=radius)
    if lw is None:
        lw = LINE_WIDTH

    p = mpatches.Circle(center, radius, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p

def add_arc_patch(ax, center, radius, theta1, theta2, facecolor='none', edgecolor=BLACK, auto_scale=True, lw=None, angle_type='rad', **kwargs):
    '''
    创建圆弧patch

    angle_type: 角度类型,默认为'rad',即弧度制.可选'rad'或'deg'
    '''
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
    p = mpatches.Arc(center, 2*radius, 2*radius, theta1=theta1, theta2=theta2, facecolor=facecolor, edgecolor=edgecolor, lw=lw, **kwargs)
    add_patch(ax, p, auto_scale=auto_scale)
    return p

def add_arrow(ax, x_start, y_start, x_end, y_end, xycoords='data', label=None, fc=RED, ec=RED, linewidth=None, arrow_sytle=ARROW_STYLE, head_width=ARROW_HEAD_WIDTH, head_length=ARROW_HEAD_LENGTH, alpha=1.0, **kwargs):
    '''
    根据起点和终点坐标绘制一个或多个箭头.实际箭头由add_annotation绘制,并额外创建空artist用于图例.支持list、array和单个数字.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param x_start: 箭头的起始x坐标.
    :param y_start: 箭头的起始y坐标.
    :param x_end: 箭头的终止x坐标.
    :param y_end: 箭头的终止y坐标.
    :param label: 箭头的标签,默认为None.
    :param fc: 箭头的填充颜色,默认为RED.
    :param ec: 箭头的边框颜色,默认为RED.
    :param linewidth: 箭头的线宽,默认为LINE_WIDTH.
    :param arrow_sytle: 箭头的样式,默认为ARROW_STYLE.
    :param head_width: 箭头的宽度,默认为ARROW_HEAD_WIDTH.
    :param head_length: 箭头的长度,默认为ARROW_HEAD_LENGTH.
    :param kwargs: 传递给`add_annotation`的额外关键字参数,同时用于创建图例artist.
    '''

    _print_invalid_value_warning('add_arrow', x_start=x_start, y_start=y_start, x_end=x_end, y_end=y_end)
    if linewidth is None:
        linewidth = LINE_WIDTH

    # 确保起点和终点坐标可以迭代
    x_start, y_start, x_end, y_end = map(
        np.atleast_1d, [x_start, y_start, x_end, y_end])
    n_arrows = len(x_start)

    for i in range(n_arrows):
        add_annotation(ax, '', (x_end[i], y_end[i]), (x_start[i], y_start[i]), xycoords=xycoords, arrowprops={
                       'fc': fc, 'ec': ec, 'linewidth': linewidth, 'arrowstyle': arrow_sytle, 'head_length': head_length, 'head_width': head_width, 'alpha': alpha}, **kwargs)

    # 为了添加label而使用的虚拟箭头
    ax.arrow([], [], [], [], label=label, fc=fc, ec=ec,
             linewidth=linewidth, head_width=0, head_length=0, **kwargs)

def add_annotation(ax, text, xy, xytext, xycoords='data', fontsize=None, arrowprops=None, text_process=None, **kwargs):
    '''
    在指定位置添加注释.只支持一个注释.
    :param ax: matplotlib的轴对象,用于绘制图形.
    :param text: 注释的文本.
    :param xy: 注释的位置.即箭头的终止位置.
    :param xytext: 注释文本的位置.即箭头的起始位置.
    :param xycoords: 注释的坐标系,默认为'data'.也可以选择'axes fraction', 'subfigure fraction'等,见https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html#matplotlib.axes.Axes.annotate.
    :param fontsize: 注释文本的字体大小,默认为FONT_SIZE.
    :param arrowprops: 箭头的属性
    :param text_process: 文本处理参数,默认为TEXT_PROCESS.
    :param kwargs: 传递给`ax.annotate`的额外关键字参数.
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    arrowprops = update_dict(ARROW_PROPS, arrowprops)
    _print_invalid_value_warning('add_annotation', xy=xy, xytext=xytext)
    if fontsize is None:
        fontsize = FONT_SIZE

    # 自动调整dict
    if 'head_length' in arrowprops.keys() and 'head_length' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_length=' + \
            str(arrowprops['head_length'])
        del arrowprops['head_length']
    if 'head_width' in arrowprops.keys() and 'head_width' not in arrowprops['arrowstyle']:
        arrowprops['arrowstyle'] += ',head_width=' + \
            str(arrowprops['head_width'])
        del arrowprops['head_width']

    if xycoords == 'data':
        # 添加一个s=0的散点,因为annotation函数不会自动调整xy轴的坐标系让箭头可见
        ax.scatter([xy[0], xytext[0]], [xy[1], xytext[1]], s=0)

    # 处理文本
    text = format_text(text, text_process)
    # 画图
    return ax.annotate(text, xy, xytext=xytext, xycoords=xycoords, fontsize=fontsize, arrowprops=arrowprops, **kwargs)

def _check_maybe_wrong_transform_for_text(x, y, transform):
    if transform != 'data':
        if x > 2 or x < -2 or y > 2 or y < -2:
            print(f"Warning: x={x}, y={y}, transform={transform} may be wrong.")

def add_text(ax, text, x=TEXT_X, y=TEXT_Y, text_process=None, transform='ax', va=TEXT_VA, ha=TEXT_HA, fontsize=None, color=BLACK, **kwargs):
    '''

    在指定位置添加文字.
    :param ax: matplotlib的轴对象,用于绘制图形
    :param text: 文本内容
    :param x: 文本的x坐标,默认为0.05
    :param y: 文本的y坐标,默认为0.95
    :param text_process: 文本处理参数,默认为TEXT_PROCESS
    :param transform: 文本的坐标系,默认为'ax'.也可以选择'fig'或'data'
    :param kwargs: 传递给'ax.text'的额外关键字参数
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    _print_invalid_value_warning('add_text', x=x, y=y)
    if fontsize is None:
        fontsize = FONT_SIZE
    _check_maybe_wrong_transform_for_text(x, y, transform)
    # 处理文本
    text = format_text(text, text_process)

    if transform == 'ax':
        transform = ax.transAxes
    elif transform == 'fig':
        transform = ax.figure.transFigure
    elif transform == 'data':
        transform = ax.transData

    return ax.text(x, y, text, transform=transform, va=va, ha=ha, fontsize=fontsize, color=color, **kwargs)

def add_multi_text_by_dict(ax, x_dict, y_dict, text_process=None, transform='ax', va=TEXT_VA, ha=TEXT_HA, fontsize=None, color=BLACK, **kwargs):
    # check the keys of x_dict and y_dict are the same
    if set(x_dict.keys()) != set(y_dict.keys()):
        raise ValueError("The keys of x_dict and y_dict must be the same.")
    text_obj_list = []
    for key in x_dict.keys():
        text_obj_list.append(add_text(ax, key, x=x_dict[key], y=y_dict[key], text_process=text_process, transform=transform, va=va, ha=ha, fontsize=fontsize, color=color, **kwargs))
    return text_obj_list

def auto_adjust_text(ax, text_obj_list, add_arrow, arrow_style='->', arrow_color=BLACK, arrow_width=0.5, expand_points=(1.2, 1.4), expand_text=(1.2, 1.4), force_points=(0.5, 0.5), force_text=(0.5, 0.5), min_arrow_len=15, ensure_inside_axes=False, seed=0, **kwargs):
    '''
    自动调整文本位置,避免重叠
    '''
    if add_arrow:
        arrow_props = dict(arrowstyle=arrow_style, color=arrow_color, lw=arrow_width)
    else:
        arrow_props = None
    adjust_text_custom.adjust_text(text_obj_list, arrowprops=arrow_props, expand_points=expand_points, expand_text=expand_text, force_points=force_points, force_text=force_text, min_arrow_len=min_arrow_len, ax=ax, ensure_inside_axes=ensure_inside_axes, seed=seed, **kwargs)

def add_ax_tag(ax, tag, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加轴的标签
    :param ax: matplotlib的轴对象,用于绘制图形
    :param tag: 标签的文本
    :param x: 标签的x坐标,默认为AX_TAG_POS[0]
    :param y: 标签的y坐标,默认为AX_TAG_POS[1]
    :param fontsize: 标签的字体大小,默认为TITLE_SIZE
    :param va: 标签的垂直对齐方式,默认为TAG_VA
    :param ha: 标签的水平对齐方式,默认为TAG_HA
    :param kwargs: 传递给'ax.text'的额外关键字参数
    '''
    # 画图
    return add_text(ax, tag, x=x, y=y, fontsize=fontsize, va=va, ha=ha, **kwargs)

def add_ax_tag_by_ref_ax(ax, tag, ref_ax, x=AX_TAG_POS[0], y=AX_TAG_POS[1], fontsize=TAG_SIZE, va=TAG_VA, ha=TAG_HA, **kwargs):
    '''
    在指定位置添加轴的标签,参考另一个轴的位置
    '''
    if x > 0:
        raise ValueError("x should be less than 0")
    if y < 1:
        raise ValueError("y should be greater than 1")
    if ref_ax is not None:
        ref_ax_width, ref_ax_height = get_ax_size(ref_ax)
        ax_width, ax_height = get_ax_size(ax)
        x = x * ref_ax_width / ax_width
        y = (y - 1) * ref_ax_height / ax_height + 1
    return add_ax_tag(ax, tag, x=x, y=y, fontsize=fontsize, va=va, ha=ha, **kwargs)

def add_axes_dict_tag(axes_dict, tag_dict=None, ref_ax=None, **kwargs):
    '''
    按照tag_dict的键值对,给axes_dict中的每个轴添加标签

    参数:
    - axes_dict: dict, 包含matplotlib的Axes对象的字典
    - tag_dict: dict or None, 标签的字典,默认为None,表示使用axes_dict的键值对(即将axes_dict的key作为标签显示)
    - ref_ax: matplotlib的Axes对象,用于参考位置,默认为None,表示不参考位置
    '''
    if tag_dict is None:
        tag_dict = {ax_name: ax_name for ax_name in axes_dict.keys()}
    for ax_name, tag in tag_dict.items():
        add_ax_tag_by_ref_ax(axes_dict[ax_name], tag, ref_ax, **kwargs)

def add_axes_list_tag_by_order(axes_list, tag_kwargs=None, ref_ax=None, **kwargs):
    '''
    按照顺序加a,b,c...标签.
    '''
    tag_kwargs = update_dict({}, tag_kwargs)
    tag_list = get_tag(len(axes_list), **tag_kwargs)
    for i, ax in enumerate(axes_list):
        add_ax_tag_by_ref_ax(ax, tag_list[i], ref_ax, **kwargs)

def plt_edge_scatter(ax, x, y, color=BLUE, edge_color=BLACK, s=None, edge_prop=1.1, label=None, density=False, edge_kwargs=None, sc_kwargs=None):
    '''
    绘制带边界的散点图.

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象,用于绘制图形.
    - x (numpy.ndarray or list): x轴的数据.
    - y (numpy.ndarray or list): y轴的数据.
    - color (str or tuple, optional): 散点的颜色,默认为BLUE.
    - edge_color (str or tuple, optional): 散点的边界颜色,默认为BLACK.
    - s (float, optional): 散点的大小,默认为MARKER_SIZE**2.
    - edge_prop (float, optional): 散点的边界比例,默认为1.1.
    - label (str or None, optional): 散点的标签,默认为None.
    - density (bool, optional): 是否绘制密度图,默认为False.
    - edge_kwargs (dict or None, optional): 传递给plt_scatter的其他参数.
    - sc_kwargs (dict or None, optional): 传递给plt_density_scatter或者plt_scatter的其他参数,取决于是否使用density.

    返回:
    - matplotlib.collections.PathCollection or None: 如果density为True,则返回plt_density_scatter的返回值,否则返回plt_scatter的返回值.
    '''
    # 设置默认参数
    if edge_kwargs is None:
        edge_kwargs = {}
    if sc_kwargs is None:
        sc_kwargs = {}
    if s is None:
        s = MARKER_SIZE**2

    # 绘制边界
    edge_size = s * edge_prop
    plt_scatter(ax, x, y, color=edge_color, s=edge_size, **edge_kwargs)

    # 绘制散点
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
    '''
    绘制颜色关于c值变化的散点图.

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象, 用于绘制图形.
    - x (numpy.ndarray or list): x轴的数据.
    - y (numpy.ndarray or list): y轴的数据.
    - c (numpy.ndarray or list): 颜色的数据.
    - cmap (matplotlib.colors.Colormap, optional): 颜色映射, 默认为CMAP.
    - norm_mode (str, optional): 颜色映射的规范化模式, 可选 'linear', 'log', 'symlog', 'two_slope'等, 默认为 'linear'
    - vmin (float, optional): 颜色映射的最小值, 默认为None.
    - vmax (float, optional): 颜色映射的最大值, 默认为None.
    - norm_kwargs (dict or None, optional): 颜色映射规范化的其他参数
    - s (float, optional): 散点的大小, 默认为MARKER_SIZE**2.
    - label (str or None, optional): 散点的标签, 默认为None.
    - label_cmap_float (float, optional): 代表性点的颜色映射模式, 默认为1.0.如果输入整数则会raise ValueError.
    - scatter_kwargs (dict or None, optional): 传递给plt_scatter的其他参数.
    - cbar (bool, optional): 是否添加颜色条,默认为True.
    - cbar_postion (str or None, optional): 颜色条的位置,默认为None.
    - cbar_kwargs (dict or None, optional): 传递给add_side_colorbar的其他参数.
    '''
    _print_invalid_value_warning('plt_colorful_scatter', c=c)
    # 设置默认参数
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if cbar_kwargs is None:
        cbar_kwargs = {}
    if s is None:
        s = MARKER_SIZE**2
    if vmin is None:
        vmin = np.nanmin(c)
    if vmax is None:
        vmax = np.nanmax(c)

    # 获取颜色映射的规范化对象
    norm = get_norm(norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 绘制散点(暂时设置label为None)
    local_scatter_kwargs = update_dict(scatter_kwargs, {'s': s})
    sc = plt_scatter(ax, x, y, color=None, c=c, cmap=cmap, label=None, norm=norm, **local_scatter_kwargs)

    # 通过label_cmap_float来绘制代表性点并作出label
    if isinstance(label_cmap_float, int):
        print('warning: "label_cmap_float" is an integer, so it will be transformed to float')
    ax.scatter([], [], color=cmap(float(label_cmap_float)), label=label, s=s)

    # 添加颜色条
    if cbar:
        if vmin > np.nanmin(c):
            cbar_kwargs['add_leq'] = True
        if vmax < np.nanmax(c):
            cbar_kwargs['add_geq'] = True
        cbars = add_side_colorbar(ax, sc, vmin=vmin, vmax=vmax, norm_mode=norm_mode, norm_kwargs=norm_kwargs, cmap=cmap, cbar_position=cbar_postion, **cbar_kwargs)
        return sc, cbars
    else:
        return sc

def plt_group_bar(ax, x, y, label_list, width=None, inner_gap=0.0, colors=CMAP, vert=True, **kwargs):
    '''
    绘制分组的柱状图.
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的分组标签,大组,每个组包含多个柱子(比如['A', 'B'])
    :param y: 一个二维列表或数组,表示每组中柱子的高度(shape=(len(x), len(label_list)),例如[[1, 2, 3], [4, 5, 6]])
    :param label_list: 每个柱子的标签,例如['x', 'y', 'z']
    :param width: 单个柱子的宽度,默认为None,自动确定宽度
    :param inner_gap: 柱子之间的间距,默认为0.0,其数值代表的是同组内柱子之间的间距占单个柱子宽度的比例
    :param colors: 柱状图的颜色序列,应与label_list的长度相匹配;也可以指定cmap,默认为CMAP,然后根据label_list的长度生成颜色序列
    :param kwargs: 其他plt.bar支持的参数
    '''
    _print_invalid_value_warning('plt_group_bar', x=x)
    # 假如colors不是一个list,则用colors对应的cmap生成颜色
    if not isinstance(colors, list):
        colors = colors(np.linspace(0, 1, len(label_list)))

    num_groups = len(y)  # 组的数量
    num_bars = len(y[0])  # 每组中柱子的数量,假设每组柱子数量相同

    if width is None:
        # Calculate width considering both number of bars and inner gaps
        # Total group width = BAR_WIDTH (default group spacing)
        # This is distributed as: num_bars * width + (num_bars - 1) * (width * inner_gap)
        # So: width = BAR_WIDTH / (num_bars + (num_bars - 1) * inner_gap)
        width = BAR_WIDTH / (num_bars + (num_bars - 1) * inner_gap)
    gap_width = width * inner_gap

    # 为每个柱子计算中心位置
    indices = np.arange(num_groups)
    for i, lbl in enumerate(label_list):
        offset = (i - (num_bars - 1) / 2) * (width + gap_width)
        plt_bar(ax, indices + offset, [y[j][i] for j in range(
            num_groups)], label=lbl, color=colors[i], width=width, vert=vert, **kwargs)

    if vert:
        ax.set_xticks(indices)
        ax.set_xticklabels(x)
    else:
        ax.set_yticks(indices)
        ax.set_yticklabels(x)

def plt_two_side_bar(ax, x, y1, y2, label1=None, label2=None, width=BAR_WIDTH, color1=BLUE, color2=RED, yerr1=None, yerr2=None, capsize=PLT_CAP_SIZE, ecolor1=BLACK, ecolor2=BLACK, elabel1=None, elabel2=None, vert=True, equal_space=False, **kwargs):
    '''
    使用x和y1,y2绘制两组柱状图
    '''
    plt_bar(ax, x, np.array(y1), label=label1, color=color1, vert=vert, equal_space=equal_space,
                 err=yerr1, capsize=capsize, ecolor=ecolor1, elabel=elabel1, width=width, **kwargs)
    plt_bar(ax, x, -np.array(y2), label=label2, color=color2, vert=vert, equal_space=equal_space,
                 err=yerr2, capsize=capsize, ecolor=ecolor2, elabel=elabel2, width=width, **kwargs)

    local_max = np.nanmax([np.nanmax(y1), np.nanmax(y2)]) * 1.2
    if vert:
        axis = 'y'
    else:
        axis = 'x'
    set_sym_positive_axis(ax, axis, local_max)

def plt_linregress(ax, x, y, xlog=False, ylog=False, xlog_base=10, ylog_base=10, xlog_base_str=None, ylog_base_str=None, linear_but_log=False, label=None, scatter_color=BLUE, line_color=RED, line_bound='data', show_list=None, round_digit_dict=None, round_format_dict=None, text_size=None, scatter_kwargs=None, line_kwargs=None, text_kwargs=None, regress_kwargs=None, show_scatter=True, scatter_mode='original'):
    '''
    使用线性回归结果绘制散点图和回归线,可以输入scatter的其他参数

    参数:
    ax - 绘图的Axes对象
    x, y - 数据
    xlog, ylog - 是否对 x 和 y 取对数
    xlog_base, ylog_base - 对数的底数
    xlog_base_str, ylog_base_str - 对数的底数的字符串形式(如果不输入则会自动从xlog_base和ylog_base中获取)
    linear_but_log - 如果为True,当x/ylog时直接将原数据log,并在线性尺度上绘制回归线,同时手动调整tick等变为x/ylog_base**k;如果为False,则在对数尺度上绘制回归线,此时ax本身是logscale的;两种方法的区别在于,对于10^0.1,10^0.2,直接使用logscale,将会显示不出来什么tick(因为logscale的tick都是整数),而手动调整tick则可以显示出来
    label - 数据标签
    x_label, y_label - 坐标轴标签
    title - 图标题
    fontsize - 标签字体大小
    x_sci, y_sci - 是否启用科学记数法
    scatter_color - 散点图颜色
    line_color - 线性回归线颜色
    show_corr - 是否显示相关系数
    show_corr_round - 相关系数小数点位数
    show_p - 是否显示 P 值
    show_p_round - P 值小数点位数
    show_scatter - 是否显示散点
    scatter_mode - 散点模式,默认为'original',可选'density'(此时scatter_kwargs中的参数会传递给plt_density_scatter)或者'edge'(此时scatter_kwargs中的参数会传递给plt_edge_scatter)
    '''
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

    # 预处理数据(如果需要对数)
    if xlog:
        x_local = np.log(x) / np.log(xlog_base)
    else:
        x_local = x.copy()
    if ylog:
        y_local = np.log(y) / np.log(ylog_base)
    else:
        y_local = y.copy()

    # 绘制散点图
    if show_scatter:
        if linear_but_log:
            if scatter_mode == 'density':
                plt_density_scatter(ax, x_local, y_local, label=label, **scatter_kwargs)
            elif scatter_mode == 'edge':
                plt_edge_scatter(ax, x_local, y_local, color=scatter_color, label=label, **scatter_kwargs)
            else:
                plt_scatter(ax, x_local, y_local, color=scatter_color, label=label, **scatter_kwargs)
        else:
            if scatter_mode == 'density':
                plt_density_scatter(ax, x, y, label=label, **scatter_kwargs)
            elif scatter_mode == 'edge':
                plt_edge_scatter(ax, x, y, color=scatter_color, label=label, **scatter_kwargs)
            else:
                plt_scatter(ax, x, y, color=scatter_color, label=label, **scatter_kwargs)

    # 设置坐标轴和tick
    if xlog:
        if linear_but_log:
            if xlog_base_str is None: # 自动获取对数底数的字符串形式
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
            if ylog_base_str is None: # 自动获取对数底数的字符串形式
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
        # 计算线性回归
        regress_dict = get_linregress(x_local, y_local, **regress_kwargs)

        # 绘制回归线
        if line_bound == 'ax':
            if xlog or ylog:
                raise ValueError('xlog and ylog cannot be True when line_bound is "ax"')
            add_sline(ax, regress_dict['slope'], regress_dict['intercept'], reference_point='auto', color=line_color, **line_kwargs)
        elif line_bound == 'data':
            x_local_min, x_local_max = np.nanmin(x_local), np.nanmax(x_local)
            # y_min不一定真的是min,这里只是为了对应x_min
            y_local_min = x_local_min * regress_dict['slope'] + regress_dict['intercept']
            y_local_max = x_local_max * regress_dict['slope'] + regress_dict['intercept']

            # 如果使用了对数坐标,将线还原到实际的 x 和 y 值
            if xlog and not linear_but_log:
                x_min, x_max = xlog_base ** x_local_min, xlog_base ** x_local_max
            else:
                x_min, x_max = x_local_min, x_local_max
            if ylog and not linear_but_log:
                y_min, y_max = ylog_base ** y_local_min, ylog_base ** y_local_max
            else:
                y_min, y_max = y_local_min, y_local_max
            plt_line(ax, [x_min, x_max], [y_min, y_max], color=line_color, **line_kwargs)

        add_text_by_dict(ax, text_dict=regress_dict, show_list=show_list, round_digit_dict=round_digit_dict, round_format_dict=round_format_dict, fontsize=text_size, **text_kwargs)
        return regress_dict

def plt_density_scatter(ax, x, y, label=None, label_cmap_float=1.0, estimate_type='kde', bw_method='auto', bins_x=BIN_NUM, bins_y=BIN_NUM, cmap=DENSITY_CMAP, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, cbar=True, cbar_label='density', cbar_position=None, scatter_kwargs=None, cbar_kwargs=None):
    '''
    绘制密度散点图.
    :param ax: matplotlib的轴对象,用于绘制图形
    :param x: x轴的数据
    :param y: y轴的数据
    :param label: 图例标签,默认为None
    :param label_cmap_float: 图例颜色,默认为1.0
    :param estimate_type: 密度估计类型,默认为'kde',可选'hist'
    :param bw_method: 密度估计的带宽方法,默认为'auto'
    :param bins_x: x轴的bins,默认为BIN_NUM
    :param bins_y: y轴的bins,默认为BIN_NUM
    :param cmap: 颜色映射,默认为DENSITY_CMAP
    :param cbar: 是否显示颜色条,默认为True
    :param scatter_kwargs: 散点图的其他参数
    :param cbar_kwargs: 颜色条的其他参数

    注意:
    如果log scale,则不太适合使用hist,因为hist的bins是线性的,而log scale的bins是对数的
    '''
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
        hist, xedges, yedges, _, _ = get_hist_2d(
            x_array, y_array, bins_x=bins_x, bins_y=bins_y, stat='density')
        ix = np.digitize(x_array, xedges, right=False) - 1
        iy = np.digitize(y_array, yedges, right=False) - 1

        ix = np.clip(ix, 0, hist.shape[0] - 1)
        iy = np.clip(iy, 0, hist.shape[1] - 1)
        z = hist[ix, iy]

    return plt_colorful_scatter(ax, x, y, c=z, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, label=label, label_cmap_float=label_cmap_float, scatter_kwargs=scatter_kwargs, cbar=cbar, cbar_postion=cbar_position, cbar_kwargs=cbar_kwargs)

def plt_smooth_scatter_line(ax, x, y, frac=0.2, scatter_label=None, line_label='LOWESS', scatter_color=BLUE, scatter_alpha=FAINT_ALPHA, line_color=RED, scatter_kwargs=None, line_kwargs=None):
    '''
    利用x和y绘制散点图和平滑线
    '''
    if scatter_kwargs is None:
        scatter_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}
    _print_invalid_value_warning('plt_smooth_scatter_line', x=x, y=y)
    x_smooth, y_smooth = lowess_smooth(x, y, frac=frac)
    return plt_scatter(ax, x, y, color=scatter_color, label=scatter_label, alpha=scatter_alpha, **scatter_kwargs), plt_line(ax, x_smooth, y_smooth, color=line_color, label=line_label, **line_kwargs)

def plt_cdf(ax, data, label=None, color=BLUE, vert=True, **kwargs):
    '''
    使用数据绘制累积分布函数图,可以接受plt.plot的其他参数
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制累积分布函数图的数据集
    :param label: 图例标签,默认为None
    :param color: 累积分布函数图的颜色,默认为BLUE
    :param kwargs: 其他plt.plot支持的参数
    '''
    _print_invalid_value_warning('plt_cdf', data=data)
    cdf = get_cdf(data)
    return plt_line(ax, cdf.x, cdf.y, label=label, color=color, vert=vert, **kwargs)

def plt_compare_cdf_ks(ax, data1, data2, label1=None, label2=None, color1=BLUE, color2=RED, vert=True, text_x=TEXT_X, text_y=TEXT_Y, fontsize=None, round_digit_dict=None, round_format_dict=None, show_list=None, text_kwargs=None, cdf_kwargs=None):
    '''
    使用两组数据绘制累积分布函数图,并计算K-S检验的p值
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data1: 第一组数据
    :param data2: 第二组数据
    :param label1: 第一组数据的标签
    :param label2: 第二组数据的标签
    :param color1: 第一组数据的颜色
    :param color2: 第二组数据的颜色
    :param kwargs: 传递给plt_cdf的额外关键字参数
    '''
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
    '''
    绘制多边形热力图.

    参数:
    - ax (matplotlib.axes.Axes): matplotlib的轴对象,用于绘制图形
    - xy_dict (dict): 包含多个区域的顶点坐标的字典.示例: {'region1': [(0, 0), (1, 1), (1, 0)], 'region2': [(1, 1), (2, 2), (2, 1)]}
    - value_dict (dict): 包含多个区域的值的字典.示例: {'region1': 1, 'region2': 2}
    - mask (dict or None, optional): 包含多个区域的掩码的字典,默认为None
    - mask_color (str or tuple, optional): 掩码的颜色,默认为MASK_COLOR
    - norm_mode (str, optional): 归一化模式,默认为'linear'
    - vmin (float or None, optional): 区域值的最小值,默认为None
    - vmax (float or None, optional): 区域值的最大值,默认为None
    - norm_kwargs (dict or None, optional): 传递给get_norm的额外关键字参数
    - cmap (str or matplotlib.colors.Colormap, optional): 区域值的颜色映射,默认为CMAP
    - edgecolor (str or tuple, optional): 区域的边框颜色,默认为BLACK
    - cbar (bool, optional): 是否显示颜色条,默认为True
    - cbar_position (str or None, optional): 颜色条的位置,默认为None
    - cbar_label (str or None, optional): 颜色条的标签,默认为None
    - fill (bool, optional): 是否填充区域(如果为False,则为区域的边框,这时可以通过linewidth参数控制边框宽度)
    - adjust_lim (bool, optional): 是否根据区域的位置和大小调整轴的限制
    - polygon_kwargs (dict or None, optional): 传递给plt_polygon的额外关键字参数
    - cbar_kwargs (dict or None, optional): 传递给add_side_colorbar的额外关键字参数

    返回:
    - add_side_colorbar的返回值,如果cbar为False,则返回None.
    '''
    # 设置默认参数
    if vmin is None:
        vmin = np.nanmin(list(value_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(value_dict.values()))
    polygon_kwargs = update_dict({}, polygon_kwargs)
    cbar_kwargs = update_dict({}, cbar_kwargs)

    # 获取use_mask
    if mask is None:
        use_mask = False
    elif isinstance(mask, dict):
        # 判断是否存在掩码
        if any(mask.values()):
            use_mask = True
        else:
            use_mask = False

    # 得到norm
    norm = get_norm(norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs)

    # 根据norm后的data值获取颜色
    for region, xy in xy_dict.items():
        if use_mask:
            if mask[region]:
                plt_polygon(ax, xy, color=None, facecolor=mask_color, edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
            else:
                plt_polygon(ax, xy, color=None, facecolor=cmap(norm(value_dict[region])), edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)
        else:
            plt_polygon(ax, xy, color=None, facecolor=cmap(norm(value_dict[region])), edgecolor=edgecolor, fill=fill, adjust_lim=adjust_lim, **polygon_kwargs)

    # 添加colorbar
    if cbar:
        return add_side_colorbar(ax, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, cbar_label=cbar_label, cbar_position=cbar_position, use_mask=use_mask, mask_color=mask_color, **cbar_kwargs)

def plt_group_bar_dict(ax, data, width=None, colors=CMAP, vert=True, **kwargs):
    """
    使用字典数据绘制分组的柱状图.

    :param ax: matplotlib的轴对象
    :param data: 字典格式输入,结构为: {'A': {'x': 1, 'y': 2, 'z': 3}, 'B': {'x': 4, 'y': 5, 'z': 6}}
    :param width: 柱子宽度 (逻辑同原函数)
    :param colors: 颜色配置 (逻辑同原函数)
    :param vert: 是否垂直柱状图
    :param kwargs: 其他plt.bar参数
    """
    # 提取分组标签 (x轴标签)
    x_labels = list(data.keys())

    # 提取子组标签并验证一致性
    subgroup_labels = list(next(iter(data.values())).keys())  # 取第一个分组的子组作为标准
    for group, subgroups in data.items():
        if set(subgroups.keys()) != set(subgroup_labels):
            raise ValueError(f"分组 '{group}' 的子组标签与首分组不一致")

    # 构造二维y值数组
    y_values = [
        [subgroups[sub] for sub in subgroup_labels]
        for groups, subgroups in data.items()
    ]

    # 调用原始函数绘图
    return plt_group_bar(ax=ax, x=x_labels, y=y_values, label_list=subgroup_labels, width=width, colors=colors, vert=vert, **kwargs)

def sns_marginal_heatmap(ax, data, x_side_ax=None, y_side_ax=None, outside=True, x_side_ax_position='top', y_side_ax_position='right', x_side_ax_pad=SIDE_PAD, y_side_ax_pad=SIDE_PAD, x_side_ax_size=0.3, y_side_ax_size=0.3, x_color=BLUE, y_color=BLUE, heatmap_kwargs=None, bar_kwargs=None, rm_tick=True, rm_spine=True, rm_axis=True, inset_mode='fig'):
    '''
    绘制带有边缘分布的热图
    '''
    heatmap_kwargs = update_dict(get_default_param(sns_heatmap), heatmap_kwargs)
    heatmap_kwargs['cbar_position'] = update_dict(CBAR_POSITION, heatmap_kwargs['cbar_position'])
    bar_kwargs = update_dict(get_default_param(plt_bar), bar_kwargs)
    bar_kwargs = update_dict(bar_kwargs, {'width': 1})

    if isinstance(data, np.ndarray):
        local_data = pd.DataFrame(data)
    else:
        local_data = data.copy()

    # 获得边缘分布需要的ax
    if x_side_ax is None and y_side_ax is None:
        if outside:
            x_side_ax = add_side_ax(ax, position=x_side_ax_position, relative_size=x_side_ax_size, pad=x_side_ax_pad, sharex=ax, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
            y_side_ax = add_side_ax(ax, position=y_side_ax_position, relative_size=y_side_ax_size, pad=y_side_ax_pad, sharey=ax, inset_mode=inset_mode, hide_repeat_xaxis=False, hide_repeat_yaxis=False)
        else:
            # 当边缘分布在内部时,需要手动分划ax然后获取需要的
            x_side_ax, y_side_ax, ax = split_with_double_marginal_ax(ax=ax, x_side_ax_position=x_side_ax_position, y_side_ax_position=y_side_ax_position, x_side_ax_size=x_side_ax_size, y_side_ax_size=y_side_ax_size, x_side_ax_pad=x_side_ax_pad, y_side_ax_pad=y_side_ax_pad, inset_mode=inset_mode)

    # 调整cbar的位置
    if heatmap_kwargs['cbar']:
        ax_width, ax_height = get_ax_size(ax)
        x_side_ax_width, x_side_ax_height = get_ax_size(x_side_ax)
        y_side_ax_width, y_side_ax_height = get_ax_size(y_side_ax)
        if heatmap_kwargs['cbar_position']['position'] == x_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += x_side_ax_pad + x_side_ax_height / ax_height
        if heatmap_kwargs['cbar_position']['position'] == y_side_ax_position:
            heatmap_kwargs['cbar_position']['pad'] += y_side_ax_pad + y_side_ax_width / ax_width

    # 绘制热图
    sns_heatmap(ax, local_data, **heatmap_kwargs)

    # 绘制边缘分布
    plt_bar(x_side_ax, x=np.arange(len(local_data.columns)) + 0.5, y=col_sum_df(local_data), **update_dict(bar_kwargs, {'color': x_color}))
    plt_bar(y_side_ax, x=np.arange(len(local_data.index)) + 0.5, y=row_sum_df(local_data), **update_dict(bar_kwargs, {'color': y_color, 'vert': False}))

    # 隐藏边缘分布的刻度和坐标轴
    if rm_tick:
        rm_ax_tick(x_side_ax)
        rm_ax_tick(y_side_ax)
    if rm_spine:
        rm_ax_spine(x_side_ax)
        rm_ax_spine(y_side_ax)
    if rm_axis:
        rm_ax_axis(x_side_ax)
        rm_ax_axis(y_side_ax)

    # 调整边缘分布的方向
    if x_side_ax_position == 'bottom':
        x_side_ax.invert_yaxis()
    if y_side_ax_position == 'left':
        y_side_ax.invert_xaxis()

    return ax, x_side_ax, y_side_ax

def add_text_by_dict(ax, text_dict, show_list=None, round_digit_dict=None, round_format_dict=None, unit_dict=None, unit_mode='()', color_dict=None, fontsize=None, text_x=TEXT_X, text_y=TEXT_Y, text_process=None, text_kwargs=None):
    '''
    利用dict的key和value添加文字(每个key一行)

    参数:
    - ax: matplotlib的轴对象,用于绘制图形
    - text_dict: dict, 文字的dict
    - show_list: list, 需要显示的key的列表
    - round_digit_dict: dict, 每个key对应的小数位数
    - round_format_dict: dict, 每个key对应的小数格式
    - unit_dict: dict, 每个key对应的单位
    - fontsize: 字体大小
    - text_x: 文字的x坐标(默认为TEXT_X)
    - text_y: 文字的y坐标(默认为TEXT_Y)
    - text_process: dict, 文字处理的dict
    - text_kwargs: dict, 文字的其他参数

    注意:
    - show_list为None时,默认显示所有的key,如果想要什么也不显示,可以传入空列表[],(但是这样的话也没有必要运行此函数)
    '''
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

    y_offset = 0.
    for key in show_list:
        if key in text_dict:
            # 格式化key
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
    '''
    根据ax的大小和num的数量,返回合适的s值,实际使用时,可以先使用这个值,然后再根据实际情况在此基础上进行调整

    注意:
    s指的是面积,并且是以点(points)的平方为单位的
    '''
    ax_width, ax_height = get_ax_size(ax)
    ax_width_point = inch_to_point(ax_width)
    ax_height_point = inch_to_point(ax_height)
    s = np.pi * (np.min([ax_width_point, ax_height_point]) / np.sqrt(num)) ** 2
    s = np.min([s, 70.])
    return s

def inch_to_point(inch):
    '''
    将plt中使用的inch单位转换为points单位
    '''
    return inch * 72

def point_to_inch(point):
    '''
    将plt中使用的points单位转换为inch单位
    '''
    return point / 72

def inch_to_ax_proportion(ax, inch, axis='x'):
    '''
    将inch单位转换为ax的比例
    '''
    ax_width, ax_height = get_ax_size(ax)
    if axis == 'x':
        return inch / ax_width
    elif axis == 'y':
        return inch / ax_height
    else:
        raise ValueError(f"Unsupported axis={axis!r}; expected 'x' or 'y'")

def point_to_ax_proportion(ax, point, axis='x'):
    '''
    将points单位转换为ax的比例
    '''
    return inch_to_ax_proportion(ax, point_to_inch(point), axis)

def get_ax_size(ax):
    '''
    获取给定轴的尺寸(以英寸为单位)

    参数:
    - ax: matplotlib轴对象

    返回:
    - ax_width: 轴的宽度(英寸)
    - ax_height: 轴的高度(英寸)
    '''
    # 获取图形的尺寸(英寸)
    fig = ax.get_figure()
    fig_width, fig_height = fig.get_size_inches()

    # 获取轴的边界框,这是相对于图形大小的
    bbox = ax.get_position()

    # 计算轴的实际尺寸(英寸)
    ax_width = fig_width * bbox.width
    ax_height = fig_height * bbox.height

    return ax_width, ax_height

@iterate_over_axs
def set_ax_face_color(ax, color):
    '''
    设置ax的背景颜色
    '''
    ax.set_facecolor(color)

@iterate_over_axs
def rm_ax_spine(ax, spines_to_remove=None):
    '''
    移除轴的边框.

    参数:
    - ax: matplotlib的Axes对象
    - spines_to_remove: 要移除的边框(边框指的仅仅是轴的线,不包括tick,ticklabel和label,如果想要移除这些,请使用rm_ax_tick,rm_ax_ticklabel,rm_ax_specific_axis)
    '''
    if spines_to_remove is None:
        spines_to_remove = ['top', 'right', 'left', 'bottom']
    if isinstance(spines_to_remove, str):
        spines_to_remove = [spines_to_remove]

    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)

@iterate_over_axs
def rm_ax_tick(ax, axis=None, which='both'):
    '''
    移除轴的刻度.

    参数:
    - ax: matplotlib的Axes对象
    - axis: 要移除的轴
    - which: 要移除的刻度, 可选 'both', 'major', 'minor'

    注意:
    - ax.set_xticks([])和ax.set_yticks([])可以移除刻度, 但是这样会在sharex和sharey的情况下移除其他轴的刻度
    '''
    if axis is None or axis == 'both':
        axis = ['x', 'y']

    for tick in axis:
        if tick == 'x':
            ax.xaxis.set_tick_params(width=0, which=which)
        elif tick == 'y':
            ax.yaxis.set_tick_params(width=0, which=which)

@iterate_over_axs
def rm_ax_ticklabel(ax, axis=None, which='both'):
    '''
    移除轴的刻度.

    参数:
    - ax: matplotlib的Axes对象
    - axis: 要移除的轴
    - which: 要移除的刻度, 可选 'both', 'major', 'minor'

    注意:
    - ax.set_xticklabels([])和ax.set_yticklabels([])会对所有sharex和sharey的轴生效, 所以不推荐使用
    '''
    if axis is None or axis == 'both':
        axis = ['x', 'y']

    for tick in axis:
        if tick == 'x':
            ax.tick_params(axis='x', which=which, labelbottom=False)
        elif tick == 'y':
            ax.tick_params(axis='y', which=which, labelleft=False)

@iterate_over_axs
def set_ax_tick(ax, ticks, labels, axis, which='major'):
    '''
    设置轴的刻度.

    参数:
    - ax: matplotlib的Axes对象
    - ticks: 刻度的位置
    - labels: 刻度的标签
    - axis: 要设置刻度的轴, 可选 'x', 'y', 'z'

    注意:
    - 一般来说,major的情形就够用了,需要设置minor的情形可能是log刻度下想要减少刻度的数量
    - 也可以直接使用set_ax函数来设置刻度,它更加自动化,还可以调整ticklabel的fontsize等,但这个函数更加轻量级
    '''
    valid_tick_types = ('major', 'minor')
    if which not in valid_tick_types:
        raise ValueError(f"which must be one of {valid_tick_types}, got {which!r}.")

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
    """
    减少指定坐标轴的刻度数量

    参数:
    ax: matplotlib的Axes对象
    axis: 要操作的轴 ('x', 'y', 'both')
    max_num: 最大刻度数量 (默认5)
    """
    if axis in ['x', 'both']:
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_num))
        ax.xaxis.set_minor_locator(plt.NullLocator())

    if axis in ['y', 'both']:
        ax.yaxis.set_major_locator(plt.MaxNLocator(max_num))
        ax.yaxis.set_minor_locator(plt.NullLocator())

def share_axis(axs, sharex=True, sharey=True):
    '''
    share_axis新版,不一定需要接收输出,可以直接在原地修改

    注意:
    如果ax1.sharex(ax2)和ax1.sharex(ax3),如果直接这么写则会报错,但是ax2.sharex(ax1)和ax3.sharex(ax1)是可以的;所以本函数有可能报错,如果遇到已经share的情况,这时候需要用户在创建时不要share,而是在后续调用本函数时share,或者调整本函数的使用顺序(如果比较难,可以使用share_axis_to_target)
    '''
    if not isinstance(sharex, str):
        sharex = "all" if sharex else "none"
    if not isinstance(sharey, str):
        sharey = "all" if sharey else "none"

    # 2d numpy array的情形,可以按行或者按列共享
    if isinstance(axs, np.ndarray):
        if axs.ndim == 2:
            nrows, ncols = axs.shape
            for i in range(nrows):
                for j in range(ncols):
                    shared_with = {"none": None, "all": axs[0, 0],
                                    "row": axs[i, 0], "col": axs[0, j]}
                    if shared_with[sharex] is not None:
                        # add_subplot是可以输入None的,但是sharex和sharey不行
                        axs[i, j].sharex(shared_with[sharex])
                    if shared_with[sharey] is not None:
                        axs[i, j].sharey(shared_with[sharey])
            return axs

    # 非2d numpy array的情形,只能按照all或者none共享
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
    # 其实不需要返回,因为是原地修改;但是为了兼容,还是返回一下;如果是单个的ax输入,get_iterable_ax会把他变成一个list,所以这里需要返回第一个
    if not isinstance(axs, plt.Axes):
        return rebuild_ax(iterable_axs, axs)
    else:
        return iterable_axs[0]

def share_axis_to_target(axs, target_ax, sharex=True, sharey=True):
    '''
    share_axis_to_target新版,不一定需要接收输出,可以直接在原地修改

    注意:
    如果ax1.sharex(ax2)和ax1.sharex(ax3),如果直接这么写则会报错,但是ax2.sharex(ax1)和ax3.sharex(ax1)是可以的;所以本函数有可能报错,如果遇到已经share的情况,这时候需要用户在创建时不要share,而是在后续调用本函数时share,或者调整本函数的使用顺序
    '''
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
    # 其实不需要返回,因为是原地修改;但是为了兼容,还是返回一下;如果是单个的ax输入,get_iterable_ax会把他变成一个list,所以这里需要返回第一个
    if not isinstance(axs, plt.Axes):
        return rebuild_ax(iterable_axs, axs)
    else:
        return iterable_axs[0]

@iterate_over_axs
def rm_ax_axis(ax):
    '''
    移除轴的坐标轴.(除了title,其他的都会被移除,比如外框,刻度,刻度标签等)

    参数:
    - ax: matplotlib的Axes对象
    '''
    ax.axis('off')

def get_xylim_by_aspect_ratio(xlim, ylim, aspect_ratio=1.0):
    """
    计算按比例的坐标轴范围

    参数:
    xlim (tuple): 原始x轴范围 (xmin, xmax)
    ylim (tuple): 原始y轴范围 (ymin, ymax)
    aspect_ratio (float): 目标宽高比 (默认1.0表示正方形比例)

    返回:
    tuple: 调整后的x轴范围
    tuple: 调整后的y轴范围
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    # 计算原始范围的中心点和范围大小
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    x_range = xmax - xmin
    y_range = ymax - ymin

    # 计算当前宽高比
    current_aspect = y_range / x_range

    # 计算目标比例因子
    target_aspect = aspect_ratio

    # 调整范围以匹配目标宽高比
    if current_aspect > target_aspect:
        # 当前y轴范围过大,需要调整x轴范围
        new_x_range = y_range / target_aspect
        new_xlim = (x_center - new_x_range/2, x_center + new_x_range/2)
        new_ylim = ylim
    else:
        # 当前x轴范围过大,需要调整y轴范围
        new_y_range = x_range * target_aspect
        new_ylim = (y_center - new_y_range/2, y_center + new_y_range/2)
        new_xlim = xlim

    return new_xlim, new_ylim

@iterate_over_axs
def rm_ax_xlabel(ax):
    ax.set_xlabel('')

@iterate_over_axs
def rm_ax_ylabel(ax):
    ax.set_ylabel('')

def label_title_process(x_label, y_label, z_label, title, text_process=None):
    '''
    处理标签和标题
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    return format_text(x_label, text_process), format_text(y_label, text_process), format_text(z_label, text_process), format_text(title, text_process)

def set_fig_title(fig, title, text_process=None, title_size=None, **kwargs):
    '''
    设置图形的标题

    参数:
    - fig: matplotlib的Figure对象
    - title: 图形的标题
    - text_process: 文本处理参数
    - title_size: 标题字体大小
    - kwargs: 传递给fig.suptitle的其他参数,比如y值,可以控制标题的位置
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    if title_size is None:
        title_size = SUP_TITLE_SIZE

    # 处理标题
    local_title = format_text(title, text_process)

    # 设置标题
    fig.suptitle(local_title, fontsize=title_size, **kwargs)

def add_axes_title(axes, title, text_process=None, title_size=None, pad=50., **kwargs):
    '''
    在多个ax的中间添加标题
    '''
    text_process = update_dict(TEXT_PROCESS, text_process)
    if title_size is None:
        title_size = TITLE_SIZE
    title = format_text(title, text_process)

    merged_ax = merge_ax(axes, rm_mode=False)
    merged_ax.set_title(title, pad=pad, fontsize=title_size, **kwargs)
    set_ax_face_color(merged_ax, color=TRANSPARENT)
    rm_ax_axis(merged_ax)

def suitable_tick_size(num_ticks, plt_size, tick_size=None, proportion=TICK_PROPORTION):
    '''
    Adjusts the font size of the tick labels based on the number of ticks and the size of the axis.

    plt_size是标量,单位inch
    '''
    if tick_size is None:
        tick_size = TICK_SIZE
    if num_ticks == 0:
        return tick_size
    suitable_tick_size = inch_to_point(plt_size) / num_ticks * proportion
    return min(suitable_tick_size, tick_size)

def get_tick_rotation(ax, axis):
    '''
    获取轴的刻度标签的旋转角度
    '''
    try:
        # 尝试获取第一个刻度标签的旋转角度
        if axis == 'x':
            return ax.xaxis.get_ticklabels()[0].get_rotation()
        elif axis == 'y':
            return ax.yaxis.get_ticklabels()[0].get_rotation()
    except IndexError:
        # 如果没有刻度标签,返回默认角度
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
    '''
    facecolor: 图例框的颜色,默认为'inherit';如果想要不设置,可以设置为'none'或者'None'(注意这个是字符串)
    edgecolor: 图例框的边框颜色,默认为'0.8';如果想要不设置,可以设置为'none'或者'None'(注意这个是字符串)
    labelcolor: label中文字的颜色,如果需要和line,scatter一致,可以设置为'linecolor'
    '''
    if fontsize is None:
        fontsize = LEGEND_SIZE
    if rm_exist_legend:
        # 去掉已有的图例
        rm_ax_legend(ax)
    # 检查是否有图例标签
    handles, labels = ax.get_legend_handles_labels()
    # 去掉重复
    handles, labels = dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys()
    if labels:
        labels = [format_text(label, text_process) for label in labels]
        legend = ax.legend(handles, labels, loc=loc,
            fontsize=fontsize, bbox_to_anchor=bbox_to_anchor, ncols=ncols, facecolor=facecolor, edgecolor=edgecolor, labelcolor=labelcolor, **kwargs)
        if frame_linewidth is None:
            frame_linewidth = LEGEND_FRAME_LINEWIDTH
        legend.get_frame().set_linewidth(frame_linewidth)

@iterate_over_axs
def set_ax(ax, xlabel=None, ylabel=None, zlabel=None, xlabel_pad=None, ylabel_pad=None, zlabel_pad=None, title=None, title_pad=None, text_process=None, title_size=None, label_size=None, tick_size=None, xtick=None, ytick=None, ztick=None, xtick_label=None, ytick_label=None, ztick_label=None, xtick_size=None, ytick_size=None, ztick_size=None, xtick_rotation=None, ytick_rotation=None, adjust_tick_size=True, tick_proportion=TICK_PROPORTION, legend=True, legend_size=None, xlim=None, ylim=None, zlim=None, xlog=False, ylog=False, zlog=False, elev=None, azim=None, legend_loc=LEGEND_LOC, bbox_to_anchor=None, rm_exist_legend=True, legend_kwargs=None, tight_layout=False, reset_scale=False):
    '''
    设置图表的轴,标题,范围和图例

    参数:
    ax - 绘图的Axes对象
    xlabel, ylabel, zlabel - x轴,y轴和z轴的标签
    xlabel_pad, ylabel_pad, zlabel_pad - x轴,y轴和z轴标签的间距
    title - 图标题
    title_pad - 标题的间距
    text_process - 文本处理参数
    title_size - 标题字体大小
    label_size - 标签字体大小
    tick_size - 刻度标签字体大小
    xtick_size, ytick_size - x轴和y轴刻度标签字体大小
    legend_size - 图例字体大小
    xlim, ylim - 坐标轴范围
    legend_loc - 图例位置
    bbox_to_anchor - 图例的位置参数(示例:(1, 1), 配合legend_loc='upper left'表示图例框的upper left角放在坐标(1, 1)处)
    tight_layout - 是否启用紧凑布局,默认为False,如果输入dict,则作为tight_layout的参数
    reset_scale - 是否在不设定xlog等的时候重新set为linear(如果True则会进行一步linear,但是这有可能破坏一些xticklabel之类的设置)
    '''
    if not isinstance(tight_layout, (bool, dict)):
        raise TypeError("tight_layout must be a bool or dict")

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

    # 尝试获取x_label和y_label
    xlabel = ax.get_xlabel() if xlabel is None else xlabel
    ylabel = ax.get_ylabel() if ylabel is None else ylabel
    if title is None:
        get_title = getattr(ax, 'get_title', None)
        title = get_title() if callable(get_title) else getattr(ax, 'title', None)
    if is_3d:
        zlabel = ax.get_zlabel() if zlabel is None else zlabel

    # 处理标签和标题
    local_xlabel, local_ylabel, local_zlabel, local_title = label_title_process(
        xlabel, ylabel, zlabel, title, text_process)

    # 设置标签和标题
    ax.set_xlabel(local_xlabel, fontsize=label_size, labelpad=xlabel_pad)
    ax.set_ylabel(local_ylabel, fontsize=label_size, labelpad=ylabel_pad)
    if is_3d:
        ax.set_zlabel(local_zlabel, fontsize=label_size, labelpad=zlabel_pad)
    ax.set_title(local_title, fontsize=title_size, pad=title_pad)

    # 设置tick
    if xtick is not None:
        set_ax_tick(ax, ticks=xtick, labels=xtick_label, axis='x')
    if ytick is not None:
        set_ax_tick(ax, ticks=ytick, labels=ytick_label, axis='y')
    if is_3d and ztick is not None:
        set_ax_tick(ax, ticks=ztick, labels=ztick_label, axis='z')

    # 设置字体
    if xtick_size is None:
        xtick_size = tick_size
    if ytick_size is None:
        ytick_size = tick_size

    if (not is_3d) and adjust_tick_size:
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

    # 设置对数坐标轴
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

    # 设置坐标轴范围
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
    '''
    添加轴的标签(方便后续直接从fig获取时,可以知道这个ax是干什么的)

    如果label是None,则设置默认标签;否则,label需要和ax的数据类型相同,且形状相同
    '''
    if isinstance(ax, dict):
        # 如果 ax 是字典
        if label is None:
            # 如果 label 为 None, 则设置默认标签
            for k, v in ax.items():
                v.set_label(k)
        elif isinstance(label, dict):
            # 如果 label 是字典
            for k in ax.keys():
                if k in label:  # 确保键在 label 中
                    ax[k].set_label(label[k])

    elif isinstance(ax, np.ndarray):
        # 如果 ax 是二维数组
        if ax.ndim == 2:
            rows, cols = ax.shape
            if label is None:
                # 如果 label 为 None, 则设置默认标签
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].set_label(f"row_{i}_col_{j}")
            elif isinstance(label, np.ndarray) and label.shape == ax.shape:
                # 如果 label 是二维数组且形状与 ax 相同
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].set_label(label[i, j])

        # 如果 ax 是一维数组
        elif ax.ndim == 1:
            length = ax.shape[0]
            if label is None:
                # 如果 label 为 None, 则设置默认标签
                for i in range(length):
                    ax[i].set_label(f"index_{i}")
            elif isinstance(label, np.ndarray) and label.shape == ax.shape:
                # 如果 label 是一维数组且形状与 ax 相同
                for i in range(length):
                    ax[i].set_label(label[i])

    elif isinstance(ax, plt.Axes):
        # 如果 ax 是单个 Axes 对象
        ax.set_label(label)

def get_suitable_fig_size(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, margin=None, adjust_params=None, adjust_params_custom=None, which=None):
    '''
        which必须显式指定为adjust_params_custom, adjust_params或margin
        adjust_params_custom 含义: 此处left, right, top, bottom的值相对于ax, 而不是fig
    '''
    valid_modes = ('adjust_params_custom', 'adjust_params', 'margin')
    if which is None:
        raise ValueError(
            f"which must be explicitly specified as one of {valid_modes}"
        )
    if which not in valid_modes:
        raise ValueError(
            f"Unsupported which={which!r}; expected one of {valid_modes}"
        )
    if which == 'adjust_params' and adjust_params is None:
        raise ValueError(
            "adjust_params must be provided when which='adjust_params'"
        )

    margin = update_dict(MARGIN, margin)
    adjust_params_custom = update_dict(ADJUST_PARAMS_CUSTOM, adjust_params_custom)
    if adjust_params_custom and which == 'adjust_params_custom':
        fig_width = ax_width * (ncols + adjust_params_custom['wspace'] * (ncols - 1)) + ax_width * (1 - adjust_params_custom['right'] + adjust_params_custom['left'])
        fig_height = ax_height * (nrows + adjust_params_custom['hspace'] * (nrows - 1)) + ax_height * (1 - adjust_params_custom['top'] + adjust_params_custom['bottom'])
    elif adjust_params and which == 'adjust_params':
        fig_width = (ax_width * (ncols + adjust_params['wspace'] * (ncols - 1))) / (adjust_params['right'] - adjust_params['left'])
        fig_height = (ax_height * (nrows + adjust_params['hspace'] * (nrows - 1))) / (adjust_params['top'] - adjust_params['bottom'])
    elif margin and which == 'margin':
        fig_width = ax_width / (margin['right'] - margin['left']) * ncols
        fig_height = ax_height / (margin['top'] - margin['bottom']) * nrows
    return fig_width, fig_height

def get_fig(width=AX_WIDTH, height=AX_HEIGHT, dpi=FIG_DPI, **kwargs):
    '''
        创建一个空白图形.
    '''
    return plt.figure(figsize=(width, height), dpi=dpi, **kwargs)

@iterate_over_subfigs
def set_subfig_size_inches(subfig, ncols, nrows):
    '''
    为 subfig 设置一个 get_size_inches 方法, 使得可以通过 subfig.get_size_inches() 获取子图的大小
    '''
    def get_size_inches(self):
        return (
            self.figure.get_size_inches()[0] / ncols,
            self.figure.get_size_inches()[1] / nrows
        )
    add_method_to_instance(subfig, get_size_inches, 'get_size_inches')

def get_subfig(fig=None, nrows=1, ncols=1, squeeze=True, wspace=None, hspace=None, width_ratios=None, height_ratios=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    subfig = fig.subfigures(nrows=nrows, ncols=ncols, squeeze=squeeze, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios, **kwargs)

    for s in get_iterable_ax(subfig):
        set_subfig_size_inches(s, ncols, nrows)
    return subfig

def get_fig_subfig(nrows=1, ncols=1, subfig_width=AX_WIDTH, subfig_height=AX_HEIGHT, fig_width=None, fig_height=None, dpi=FIG_DPI, get_fig_kwargs=None, subfig_kwargs=None):
    '''
        创建一个图形和子图对象.

        利用subfig有几个好处:
        可以创建不均匀的ax,比如说左侧三个,右侧两个ax(先创建两个subfig然后利用get_ax分别创建ax,记得设置adjust_params或者adjust_params_custom而不是margin使得边框对齐)
        可以更好的share_axis(比如说画两列图,左侧的图sharex,右侧的图sharey,这样就可以分别设置sharex和sharey)
        可以更好的set_fig_title

        利用subfig的缺点:
        无法很好的控制ax_width和ax_height,因为fig_width和fig_height由subfig_width和subfig_height决定(需要在get_ax之后,根据subfig的数量和每个subfig的大小(利用get_suitable_fig_size)来手动重制fig的大小(set_fig_size),或者在创建时计算好fig的大小,利用get_suitable_fig_size)
    '''
    get_fig_kwargs = update_dict({}, get_fig_kwargs)
    subfig_kwargs = update_dict({}, subfig_kwargs)
    adjust_params = {'left': 0., 'right': 1., 'top': 1., 'bottom': 0., 'wspace': 0., 'hspace': 0.}
    adjust_params = update_dict(adjust_params, subfig_kwargs)
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows, ncols, subfig_width, subfig_height, adjust_params=adjust_params, which='adjust_params')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height

    fig = get_fig(fig_width, fig_height, dpi=dpi, **get_fig_kwargs)
    subfig = get_subfig(nrows=nrows, ncols=ncols, fig=fig, **subfig_kwargs)
    return fig, subfig

def get_ax(fig=None, nrows=1, ncols=1, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, margin=None, squeeze=True, label='ax', subplots_params=None):
    '''
        在一个fig上创建多个ax

        fig: 可以是fig也可以是subfig,如果是None,则使用plt.gcf()获取当前的fig
        nrows, ncols: 行数和列数
        sharex, sharey: 是否共享x轴和y轴(可以是'all', 'none', 'row', 'col', True, False)
        rm_repeat_tick_label_when_share: 是否移除多余的tick label(当sharex或者sharey时)
        margin: 边框空白大小(默认为None,即使用MARGIN)
        squeeze: 是否压缩(nrows, ncols)中的1维
        adjust_params: 字典,包含用于调用subplots_adjust的参数(不推荐使用,因为随着子图的数量的增加,这个参数会变得很难调整,因为这里的left,right,top,bottom是相对于整个fig的位置,而随着ax数量增加,fig的大小会变化)(如果要使用这个参数,推荐使用get_fig_gs,然后从gs获取ax,或者创建好ax之后,使用adjust_ax)
        adjust_params_custom: 字典,含义: 此处left, right, top, bottom的值相对于ax, 而不是fig(推荐使用,不受子图数量的影响,但无法保证不同子图数量情形保证边框对齐)(如果要使用这个参数,推荐使用get_fig_gs,然后从gs获取ax,或者创建好ax之后,使用adjust_ax_custom)

        注意: 在某个fig或者subfig上创建ax,使用ax.figure或者fig.get_figure()都可以得到这个fig或者subfig(而不是最一开始最大的fig)
    '''
    if fig is None:
        fig = plt.gcf()
    margin = update_dict(MARGIN, margin)
    subplots_params = update_dict({}, subplots_params)

    if not isinstance(sharex, str):
        sharex = "all" if sharex else "none"
    if not isinstance(sharey, str):
        sharey = "all" if sharey else "none"

    ax = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            shared_with = {"none": None, "all": ax[0, 0],
                            "row": ax[i, 0], "col": ax[0, j]}
            ax[i, j] = fig.add_subplot(nrows, ncols, i * ncols + j + 1, sharex=shared_with[sharex], sharey=shared_with[sharey], **subplots_params)

    if rm_repeat_tick_label_when_share:
        # 移除多余的label
        if sharex in ["col", "all"]:
            for a in ax.flat:
                a._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ["row", "all"]:
            for a in ax.flat:
                a._label_outer_yaxis(skip_non_rectangular_axes=True)

    # 设置ax的位置
    set_relative_ax_position(ax, nrows, ncols, margin=margin)

    if squeeze:
        ax = squeeze_ax(ax)

    # 添加label
    set_ax_label(ax)

    # 略微修改label
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax

def get_fig_ax(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, subplots_params=None, squeeze=True, margin=None, label='ax'):
    '''
    创建Figure和等大小的Axes.

    默认根据nrows, ncols, ax_width, ax_height和margin计算Figure尺寸. 显式提供fig_width或fig_height时, 对应尺寸会覆盖自动计算结果. margin表示Axes在每个网格单元内的位置比例, 并传递给get_ax.

    如果需要相对于整个Figure设置adjust_params, 请使用get_fig_gs_original. 如果需要相对于单个Axes设置adjust_params_custom, 请使用get_fig_gs_custom. 如果需要不等大小的Axes, 请使用GridSpec, subfigure, merge_ax或split_ax.

    参数:
    - nrows, ncols: Axes的行数和列数.
    - ax_width, ax_height: 自动计算Figure尺寸时使用的单个Axes宽度和高度.
    - fig_width, fig_height: 可选的Figure宽度和高度, 单位为inch.
    - sharex, sharey: Axes共享坐标轴的方式.
    - rm_repeat_tick_label_when_share: 共享坐标轴时是否移除重复刻度标签.
    - subplots_params: 传递给Figure.add_subplot的参数.
    - squeeze: 是否压缩返回的Axes数组维度.
    - margin: Axes在每个网格单元内的位置比例.
    - label: Axes标签的前缀.

    返回:
    - fig: 创建的Figure.
    - ax: 创建的Axes或Axes数组.
    '''
    subplots_params = update_dict({}, subplots_params)
    margin = update_dict(MARGIN, margin)

    # 计算fig的宽度和高度
    local_fig_width, local_fig_height = get_suitable_fig_size(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, margin=margin, which='margin')
    if fig_width is None:
        fig_width = local_fig_width
    if fig_height is None:
        fig_height = local_fig_height

    # 创建图形和轴对象
    fig = get_fig(width=fig_width, height=fig_height)
    ax = get_ax(fig=fig, nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, rm_repeat_tick_label_when_share=rm_repeat_tick_label_when_share, margin=margin, squeeze=squeeze, label=label, subplots_params=subplots_params)
    return fig, ax

def gfa(nrows=1, ncols=1, ax_width=AX_WIDTH, ax_height=AX_HEIGHT, fig_width=None, fig_height=None, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, subplots_params=None, squeeze=True, margin=None, label='ax'):
    '''
    get_fig_ax的缩写
    '''
    return get_fig_ax(nrows=nrows, ncols=ncols, ax_width=ax_width, ax_height=ax_height, fig_width=fig_width, fig_height=fig_height, sharex=sharex, sharey=sharey, rm_repeat_tick_label_when_share=rm_repeat_tick_label_when_share, subplots_params=subplots_params, squeeze=squeeze, margin=margin, label=label)

def get_gs_original(fig=None, nrows=1, ncols=1, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    获取一个GridSpec对象.(不推荐使用,因为随着子图的数量的增加,这个参数会变得很难调整,因为这里的left,right,top,bottom是相对于整个fig的位置,而随着ax数量增加,fig的大小会变化,对于相同的值边框空白大小会不同,无法对齐)

    注意:
    对于gs,其始终是2维的,即使是1行或者1列
    '''
    # 如果没有提供figure,则使用当前的图形
    if fig is None:
        fig = plt.gcf()
    return GridSpec(nrows, ncols, figure=fig, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)

def get_gs_inside_ax(ax, nrows=1, ncols=1, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
    '''
    以ax的位置为基础获取一个GridSpec对象.
    '''
    left, right, bottom, top = get_ax_position_custom(ax)
    fig = ax.figure
    return get_gs_original(fig=fig, nrows=nrows, ncols=ncols, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios)

def get_all_ax_from_gs(gs, sharex=False, sharey=False, rm_repeat_tick_label_when_share=RM_REPEAT_TICK_LABEL_WHEN_SHARE, squeeze=True, label='gs', **kwargs):
    '''
    从GridSpec对象中获取所有的轴对象.
    '''
    if not isinstance(sharex, str):
        sharex = "all" if sharex else "none"
    if not isinstance(sharey, str):
        sharey = "all" if sharey else "none"

    ax = np.empty((gs.nrows, gs.ncols), dtype=object)
    for i in range(gs.nrows):
        for j in range(gs.ncols):
            shared_with = {"none": None, "all": ax[0, 0],
                            "row": ax[i, 0], "col": ax[0, j]}
            ax[i, j] = get_ax_from_gs(gs, index=(i, j), sharex=shared_with[sharex], sharey=shared_with[sharey], **kwargs)

    if rm_repeat_tick_label_when_share:
        # 移除多余的label
        if sharex in ["col", "all"]:
            for a in ax.flat:
                a._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ["row", "all"]:
            for a in ax.flat:
                a._label_outer_yaxis(skip_non_rectangular_axes=True)

    if squeeze:
        ax = squeeze_ax(ax)

    # 添加label
    set_ax_label(ax)

    # 略微修改label
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax

def get_ax_from_gs(gs, index=None, label='gs', **kwargs):
    '''
    从GridSpec对象中获取轴对象.

    参数:
    gs: 可以是完整的gs,也可以是索引之后的gs(如果是索引之后的gs,则index必须为None)
    index: 默认为None,表示获取整个gs的ax,如果是tuple,则表示获取gs[index]的ax;注意,完整的gs无法直接用于获取ax,需要使用[:]索引才可以
    '''
    fig = gs.figure
    if index is None:
        try:
            ax = fig.add_subplot(gs[:], **kwargs) # 使用gs,如果是原始的gs,必须要索引,否则会报错
        except:
            ax = fig.add_subplot(gs, **kwargs) # 为了防止用户传入索引之后的gs,这里使用gs再次尝试
    else:
        if not isinstance(index, tuple):
            raise ValueError('Index must be a tuple.')
        ax = fig.add_subplot(gs[index], **kwargs)

    # 添加label
    set_ax_label(ax, index)

    # 略微修改label
    for a in get_iterable_ax(ax):
        set_ax_label(a, cat(label, a.get_label()))
    return ax

def save_fig(fig, filename, formats=None, dpi=SAVEFIG_DPI, close=True, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES, filename_process=None, pkl=SAVEFIG_PKL, ax=None, **kwargs):
    '''
    保存图形到指定的文件格式(搭配concat_str使用,concat_str可以用于生成文件名)

    参数:
    filename - 保存文件的基础名(不建议包含扩展名)
    formats - 要保存的文件格式列表,默认为 [SAVEFIG_FORMAT]
    dpi - 图像的分辨率,默认为 SAVEFIG_DPI
    close - 是否在保存后关闭图形,默认为 True
    bbox_inches - 设置边界框(bounding box)尺寸,默认为 BBOX_INCHES
    '''
    if formats is None:
        formats = [SAVEFIG_FORMAT]
    else:
        formats = list(formats)
    filename_process = update_dict(FILENAME_PROCESS, filename_process)

    # 对filename进行处理
    filename = format_filename(filename, filename_process)

    # 从filename中获取文件夹名并创建文件夹
    mkdir(os.path.dirname(filename))

    if filename.endswith(('.png', '.pdf', '.eps')):
        # 假如filename以'.png','.pdf'或'.eps'结尾,将后缀名添加到保存的格式列表中
        formats.append(filename.split('.')[-1])
        filename = filename[:-4]

    # 按照formats列表中的格式保存
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=dpi,
                    bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

    # 保存fig到pkl
    if pkl:
        save_pkl((fig, ax), filename)

    if close:
        plt.close(fig)
        fig = None

class SingleVisualizer:
    '''
    可视化单个对象

    save_fig_kwargs的调整可以方便的影响每次的save_fig,方便在测试时使用png和低dpi,而在正式时使用pdf和高dpi
    '''
    def __init__(self, title, save_fig_kwargs=None):
        self.title = title
        self.update_save_fig_kwargs(save_fig_kwargs)

    def get_fig_ax(self, **kwargs):
        self.fig, self.ax = get_fig_ax(**kwargs)
        return self.fig, self.ax

    def update_save_fig_kwargs(self, save_fig_kwargs):
        '''
        更新save_fig_kwargs
        '''
        if not hasattr(self, 'save_fig_kwargs'):
            self.save_fig_kwargs = {}
        self.save_fig_kwargs = update_dict(self.save_fig_kwargs, save_fig_kwargs)

    def save_fig(self, filename, save_fig_kwargs=None, fig=None):
        '''
        保存fig
        '''
        if fig is None:
            fig = self.fig
        save_fig_kwargs = update_dict(self.save_fig_kwargs, save_fig_kwargs)
        save_fig(fig, filename, **save_fig_kwargs)

class MultiVisualizer(SingleVisualizer):
    '''
    可视化多个对象

    MultiVisualizer本身也可以作为一个SingleVisualizer,从而可以嵌套使用
    例如:
    每个脑区拥有一个SingleVisualizer
    全脑拥有多个脑区的SingleVisualizer,成为MultiVisualizer
    全脑的不同参数,可以以全脑的MultiVisualizer作为一个SingleVisualizer
    '''
    def __init__(self, single_visualizer_list, save_fig_kwargs=None, fig_title_list=None, title=None):
        '''
        注意:
        fig_title_list: 用于设置fig的标题,如果不设置,则使用title_list作为标题(所以它可以重复),因为仅仅用于图片内,而不会用于索引single_visualizer_list
        '''
        self.title_list = [single_visualizer.title for single_visualizer in single_visualizer_list]
        self.single_visualizer_list = single_visualizer_list.copy()
        self.set_fig_title_list(fig_title_list)
        super().__init__(title=title, save_fig_kwargs=save_fig_kwargs)

    def update_save_fig_kwargs(self, save_fig_kwargs):
        '''
        更新save_fig_kwargs

        注意:
        会同步更新所有的single_visualizer的save_fig_kwargs
        '''
        super().update_save_fig_kwargs(save_fig_kwargs)
        for single_visualizer in self.single_visualizer_list:
            single_visualizer.update_save_fig_kwargs(save_fig_kwargs)

    @property
    def num(self):
        return len(self.single_visualizer_list)

    def set_fig_title_list(self, fig_title_list=None):
        '''
        设置fig的标题(有时我们不希望使用title_list作为标题)
        '''
        if fig_title_list is None:
            self.fig_title_list = self.title_list.copy()
        else:
            self.fig_title_list = fig_title_list.copy()

    def get_single_visualizer(self, title):
        '''
        获取单个visualizer
        '''
        for single_visualizer in self.single_visualizer_list:
            if single_visualizer.title == title:
                return single_visualizer

    @contextlib.contextmanager
    def temporary_disable(self, titles):
        '''
        利用with语句临时移除指定的visualizer

        示例:
        with multi_visualizer.temporary_disable(['title1', 'title2']):
            multi_visualizer.get_fig_subfig_ax(...)
            multi_visualizer.add_tag()
        '''
        try:
            # 临时移除指定 visualizer
            self._original_single_visualizer_list = self.single_visualizer_list.copy()
            self._original_title_list = self.title_list.copy()
            self._original_fig_title_list = self.fig_title_list.copy()

            rm_idx = []
            for title in titles:
                idx = self.title_list.index(title)
                rm_idx.append(idx)
            self.single_visualizer_list = [single_visualizer for i, single_visualizer in enumerate(self.single_visualizer_list) if i not in rm_idx]
            self.title_list = [title for i, title in enumerate(self.title_list) if i not in rm_idx]
            self.fig_title_list = [fig_title for i, fig_title in enumerate(self.fig_title_list) if i not in rm_idx]
            yield
        finally:
            # 恢复原始列表
            self.single_visualizer_list = self._original_single_visualizer_list.copy()
            self.title_list = self._original_title_list.copy()
            self.fig_title_list = self._original_fig_title_list.copy()

    @classmethod
    def from_multi_visualizer_instance(cls, multi_visualizer_instance):
        '''
        从实例化的MultiVisualizer设定参数

        示例:
        new_multi_visualizer = MultiVisualizer.from_multi_visualizer_instance(multi_visualizer)
        '''
        return cls(
            copy.deepcopy(multi_visualizer_instance.single_visualizer_list),
            copy.deepcopy(multi_visualizer_instance.save_fig_kwargs),
            copy.deepcopy(multi_visualizer_instance.fig_title_list),
            copy.deepcopy(multi_visualizer_instance.title),
        )

    def get_fig_subfig_ax(self, subfig_nrows='auto', subfig_ncols='auto', separate=False, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
        '''
        获取fig和ax

        参数:
        subfig_nrows, subfig_ncols: 行数和列数(对于subfig而言,如果separate为True,则nrows和ncols无效)
        separate: True,则每个对象一个fig;False,每个对象一个subfig,但是所有对象一个fig
        '''
        if subfig_nrows == 'auto' and subfig_ncols == 'auto':
            raise ValueError('subfig_nrows and subfig_ncols cannot be both auto.')

        if subfig_nrows == 'auto':
            subfig_nrows = int(np.ceil(self.num / subfig_ncols))
        if subfig_ncols == 'auto':
            subfig_ncols = int(np.ceil(self.num / subfig_nrows))

        self.subfig_nrows = subfig_nrows
        self.subfig_ncols = subfig_ncols
        self.separate = separate
        self.subfig_dict = {}
        self.ax_dict = {}
        get_suitable_fig_size_kwargs = update_dict({}, get_suitable_fig_size_kwargs)
        get_fig_subfig_kwargs = update_dict({}, get_fig_subfig_kwargs)
        get_ax_kwargs = update_dict({}, get_ax_kwargs)
        self.ax_nrows = get_ax_kwargs.get('nrows', 1)
        self.ax_ncols = get_ax_kwargs.get('ncols', 1)

        subfig_width, subfig_height = get_suitable_fig_size(**get_suitable_fig_size_kwargs)

        if separate:
            for title in self.title_list:
                self.subfig_dict[title] = get_fig(subfig_width, subfig_height)
                self.ax_dict[title] = get_ax(fig=self.subfig_dict[title], **get_ax_kwargs)
        else:
            self.fig, subfig = get_fig_subfig(nrows=subfig_nrows, ncols=subfig_ncols, subfig_width=subfig_width, subfig_height=subfig_height, **get_fig_subfig_kwargs)
            if self.num == 1:
                subfig = np.array([[subfig]])
            subfig = subfig.flatten()
            for i, title in enumerate(self.title_list):
                self.subfig_dict[title] = subfig[i]
                self.ax_dict[title] = get_ax(fig=self.subfig_dict[title], **get_ax_kwargs)

        return self.subfig_dict, self.ax_dict


    def add_tag(self, **kwargs):
        '''
        添加tag

        注意:
        如果希望以自定义的顺序添加tag,可以自行使用add_axes_list_tag_by_order
        '''
        if self.separate:
            for ax in self.ax_dict.values():
                add_axes_list_tag_by_order(np.array(ax).flatten(), **kwargs)
        else:
            ax_in_order = []
            for title in self.title_list:
                ax_in_order.append(self.ax_dict[title])
            ax_in_order = flatten_list(pure_list(ax_in_order))
            add_axes_list_tag_by_order(ax_in_order, **kwargs)

    def add_subfig_title(self, **kwargs):
        '''
        为每个subfig添加标题
        '''
        for i in range(self.num):
            subfig = self.subfig_dict[self.title_list[i]]
            fig_title = self.fig_title_list[i]
            set_fig_title(subfig, fig_title, **kwargs)

    def rm_redundant_xylabel(self):
        pass


    def multi_ax_plot(self, func_list, kwargs_list):
        '''
        在多个ax上作图
        '''
        for func, kwargs in zip(func_list, kwargs_list):
            func(**kwargs)

    def multi_ax_plot_by_single_visualizer(self, func_name, kwargs_list):
        '''
        通过SingleVisualizer的方法在多个ax上作图
        '''
        func_list = [getattr(single_visualizer, func_name) for single_visualizer in self.single_visualizer_list]
        self.multi_ax_plot(func_list, kwargs_list)

    def save_fig(self, filename, save_fig_kwargs=None):
        '''
        保存fig
        '''
        save_fig_kwargs = update_dict(self.save_fig_kwargs, save_fig_kwargs)
        if self.separate:
            for title in self.title_list:
                save_fig(self.subfig_dict[title], filename+'_'+title, **save_fig_kwargs)
        else:
            save_fig(self.fig, filename, **save_fig_kwargs)

class MultiVisualizerByRow(MultiVisualizer):
    '''
    快捷的可视化多个对象,按行排列

    注意:
    仅仅支持每个subfig内部的ax只有一行
    '''
    def __init__(self, single_visualizer_list, save_fig_kwargs=None, fig_title_list=None, title=None):
        super().__init__(single_visualizer_list, save_fig_kwargs, fig_title_list, title)

    def rm_redundant_xylabel(self):
        '''
        去掉多余的xy标签
        '''
        if self.separate:
            pass
        else:
            for row, ax in enumerate(self.ax_dict.values()):
                if row != self.ax_nrows-1:
                    rm_ax_xlabel(ax)

    def get_fig_subfig_ax(self, subfig_ncols=1, separate=False, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
        '''
        使用父类,但是自动设定nrows
        '''
        return super().get_fig_subfig_ax(self.num, subfig_ncols, separate, get_suitable_fig_size_kwargs, get_fig_subfig_kwargs, get_ax_kwargs)

class MultiVisualizerByCol(MultiVisualizer):
    '''
    快捷的可视化多个对象,按列排列

    注意:
    仅仅支持每个subfig内部的ax只有一列
    '''
    def __init__(self, single_visualizer_list, save_fig_kwargs=None, fig_title_list=None, title=None):
        super().__init__(single_visualizer_list, save_fig_kwargs, fig_title_list, title)

    def rm_redundant_xylabel(self):
        '''
        去掉多余的xy标签
        '''
        if self.separate:
            pass
        else:
            for col, ax in enumerate(self.ax_dict.values()):
                if col != 0:
                    rm_ax_ylabel(ax)

    def add_tag(self, row_first=True, **kwargs):
        '''
        添加tag

        注意:
        row_first为True,则按行添加tag;False,则按列添加tag
        row_first要求subfig内部的ax只有一列
        '''
        if row_first:
            if self.separate:
                for ax in self.ax_dict.values():
                    add_axes_list_tag_by_order(ax, **kwargs)
            else:
                ax_in_order = []
                for title in self.title_list:
                    ax_in_order.append(self.ax_dict[title])
                ax_in_order = np.array(ax_in_order).T.flatten()
                add_axes_list_tag_by_order(ax_in_order, **kwargs)
        else:
            super().add_tag(**kwargs)

    def get_fig_subfig_ax(self, subfig_nrows=1, separate=False, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
        '''
        使用父类,但是自动设定ncols
        '''
        return super().get_fig_subfig_ax(subfig_nrows, self.num, separate, get_suitable_fig_size_kwargs, get_fig_subfig_kwargs, get_ax_kwargs)

def sns_hist(ax, data, bins=BIN_NUM, label=None, color=BLUE, log_scale=False, stat='probability', vert=True, **kwargs):
    '''
    使用数据绘制直方图,可以接受sns.histplot的其他参数(推荐使用)
    :param ax: matplotlib的轴对象,用于绘制图形
    :param data: 用于绘制直方图的数据(一维数组或列表)
    :param bins: 直方图的箱数,默认为None,自动确定箱数
    :param label: 图例标签,默认为None
    :param color: 直方图的颜色,默认为BLUE
    :param log_scale: 是否使用对数刻度,默认为False
    :param stat: 统计类型,默认为'probability'.'count': show the number of observations in each bin;'frequency': show the number of observations divided by the bin width;'probability' or 'proportion': normalize such that bar heights sum to 1;'percent': normalize such that bar heights sum to 100;'density': normalize such that the total area of the histogram equals 1
    :param vert: 是否垂直显示,默认为True
    :param kwargs: 其他sns.histplot支持的参数
    '''
    # 画图
    if vert:
        sns.histplot(data, bins=bins, label=label, color=color, ax=ax, log_scale=log_scale, stat=stat, **kwargs)
    else:
        local_data = pd.DataFrame({'temp': data})
        original_ylabel = ax.get_ylabel()
        sns.histplot(local_data, y='temp', bins=bins, label=label, color=color, ax=ax, log_scale=log_scale, stat=stat, **kwargs)
        ax.set_ylabel(original_ylabel)
