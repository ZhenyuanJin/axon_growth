import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import polygonize, unary_union
import random

def generate_random_polygon(num_vertices=6, radius=1):
    """生成随机多边形"""
    angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
    points = []
    for angle in angles:
        r = radius * (0.8 + 0.4 * np.random.random())  # 添加一些随机性
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        points.append((x, y))
    
    # 确保多边形是闭合的
    polygon = Polygon(points)
    return polygon

def create_split_lines(bounds, line_type, num_lines):
    """创建分割线"""
    lines = []
    min_x, min_y, max_x, max_y = bounds
    
    if line_type == 'horizontal':
        for i in range(1, num_lines + 1):
            y = min_y + i * (max_y - min_y) / (num_lines + 1)
            # 延长线确保与多边形相交
            line = LineString([(min_x - 1, y), (max_x + 1, y)])
            lines.append(line)
    
    elif line_type == 'vertical':
        for i in range(1, num_lines + 1):
            x = min_x + i * (max_x - min_x) / (num_lines + 1)
            # 延长线确保与多边形相交
            line = LineString([(x, min_y - 1), (x, max_y + 1)])
            lines.append(line)
    
    return lines

def split_polygon_with_lines(polygon, lines):
    """使用一组线分割多边形"""
    result_polygons = [polygon]
    
    for line in lines:
        new_polygons = []
        for poly in result_polygons:
            if line.crosses(poly):
                try:
                    # 使用缓冲技巧进行分割
                    buffered_line = line.buffer(0.001)
                    difference = poly.difference(buffered_line)
                    
                    if difference.geom_type == 'Polygon':
                        new_polygons.append(difference)
                    elif difference.geom_type == 'MultiPolygon':
                        new_polygons.extend(list(difference.geoms))
                    else:
                        new_polygons.append(poly)
                except:
                    new_polygons.append(poly)
            else:
                new_polygons.append(poly)
        
        result_polygons = new_polygons
    
    return [p for p in result_polygons if isinstance(p, Polygon) and not p.is_empty]

def split_polygon_grid(polygon, rows, cols):
    """使用网格线分割多边形"""
    bounds = polygon.bounds
    
    # 创建网格线
    horizontal_lines = create_split_lines(bounds, 'horizontal', rows - 1)
    vertical_lines = create_split_lines(bounds, 'vertical', cols - 1)
    
    # 组合所有线
    all_lines = horizontal_lines + vertical_lines
    
    # 使用通用的分割函数
    return split_polygon_with_lines(polygon, all_lines)