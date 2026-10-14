

import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pointpats
import scipy
import scipy.stats as st
import shapely.geometry
import SimpleITK as sitk
from scipy.spatial.distance import pdist
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm

from . import cf
from .model_runtime import (ConnectionModel, load_connectivity,
                            save_connectivity, save_json)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COLOR = cf.RED
TARGET_COLOR = cf.GREEN
DATA_COLOR = cf.PURPLE
MODEL_COLOR = cf.GREEN

OPTIMAL_PARAMS = {
    'mouse_gamanut': {'Lambda': 0.85},
    'marmoset_majka': {'Lambda': 2.075},
    'macaque_markov40': {'Lambda': 9.6},
    'human_lariviere': {'Lambda': 22.63},
}
OPTIMAL_PARAMS['marmoset_V4'] = OPTIMAL_PARAMS['marmoset_majka'].copy()




def coords_to_polygon(x, y):
    return shapely.geometry.Polygon(np.column_stack((x, y)))


def organize_txt_coords(filename):
    name_position_start = str(filename).rfind('/')
    name_position_end = str(filename).find('.txt')
    name = str(filename)[name_position_start+1: name_position_end]

    txt_content = cf.load_txt(filename)
    txt_content = [l for l in txt_content if l != '']
    txt_content = [l.split() for l in txt_content]
    txt_content = [[float(item) for item in l] for l in txt_content]
    txt_content = np.array(txt_content)
    vx, vy = txt_content[:, 0], txt_content[:, 1]

    coords = np.array([np.array([i,j]) for i, j in zip(vx, vy)])
    polygon = coords_to_polygon(vx, vy)
    results = {'coords': coords, 'polygon': polygon, 'name': name}
    return results


def generate_coords_polygon(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)

    cf.mkdir(animal_dir.area_coords_dir)


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


    for filename in cf.get_subfile(animal_dir.raw_whole_cortex_coords_dir):
        r = organize_txt_coords(filename)
        whole_cortex_coords, whole_cortex_polygon = r['coords'], r['polygon']
    cf.save_pkl(whole_cortex_coords, animal_dir.area_coords_dir + 'whole_cortex_coords')
    cf.save_pkl(whole_cortex_polygon, animal_dir.area_coords_dir + 'whole_cortex_polygon')
    results = {'coords_dict': coords_dict, 'polygon_dict': polygon_dict, 'whole_cortex_coords': whole_cortex_coords, 'whole_cortex_polygon': whole_cortex_polygon}
    return results


def generate_area_center(animal, dataset_key, density=100):
    animal_info = AnimalInfo(animal, dataset_key)
    polygon_dict = animal_info.polygon_dict


    seed = 0

    area_center_dict = {}
    for area, polygon in polygon_dict.items():
        points = get_random_points_in_polygon(polygon=polygon, density=density, seed=seed)
        cx, cy = np.mean(points[:, 0]), np.mean(points[:, 1])
        area_center_dict[area] = np.array([cx,cy])
    cf.save_dict(area_center_dict, animal_info.area_coords_dir + 'area_center_dict')
    return area_center_dict


def get_area_voxel_3D(animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    full_name = animal_info.full_name
    area_voxel_dict = {}
    if animal=='mouse' and dataset_key=='gamanut':
        atlas = np.load(animal_info.raw_3D_data_dir+'/modified_data.npy')
        areas = pd.read_excel(animal_info.raw_3D_data_dir+'/areas.xlsx')
        modified_area_to_index = {}
        for idx in areas.index:
            modified_area_to_index[areas.loc[idx, 'area_name']] = idx + 1
        for brain_area in full_name:

            s = modified_area_to_index[brain_area]
            voxels = np.array(np.where(atlas==s)).T * 0.025
            area_voxel_dict[brain_area] = voxels
    if animal=='marmoset' and dataset_key=='majka':
        def indices_to_mm(repeated_indices, affine):
            return np.array([repeated_indices[:, 0]*(-affine[0, 0])-affine[0, 3], repeated_indices[:, 1]*(-affine[1, 1])-affine[1, 3], repeated_indices[:, 2]*affine[2, 2]+affine[2, 3]]).T
        atlas = nib.load(animal_info.raw_3D_data_dir+'/atlas_segmentation_200um.nii.gz')
        area_name = pd.read_excel(animal_info.raw_3D_data_dir+'/atlas_labels.xlsx')
        atlas_label = atlas.get_fdata()
        affine = atlas.affine
        for label in np.unique(atlas_label):
            if int(label)!=0:
                voxel = np.array(np.where(atlas_label==label)).T
                area_voxel_dict[area_name.loc[(int(label))-1, 'Area']] = indices_to_mm(voxel, affine)
        area_voxel_dict.pop('no', None)
    if animal=='macaque' and dataset_key in ['markov29', 'markov40']:
        def macaque_voxel_value_to_area_name(voxel_value):
            label_mapping = {0: 'Clear Label', 1: '10', 2: '46d', 3: '46v', 4: '11', 5: '12', 6: '14', 7: '9', 8: '32', 9: '946d', 10: '946v', 11: '24c', 12: '8B', 13: '45A', 14: '13', 15: '25', 16: 'TEMPORAL_POLE', 17: '45B', 18: 'F6', 19: 'F7', 20: '8r', 21: 'ProM', 22: '44', 23: 'F5', 24: '24b', 25: 'OPAI', 26: 'OPRO', 27: 'Gu', 28: '8l', 29: '8m', 30: '24a', 31: 'PERIRHINAL', 32: 'PIRIFORM', 33: 'ENTORHINAL', 34: 'STPr', 35: 'TEama', 36: 'TEav', 37: 'INSULA', 38: 'Pi', 39: 'PBr', 40: 'IPa', 41: 'SII', 42: 'PGa', 43: 'CORE', 44: 'LB', 45: 'MB', 46: '24d', 47: 'TEad', 48: 'F2', 49: 'F3', 50: 'F4', 51: '3', 52: '2', 53: 'F1', 54: 'SUBICULUM', 55: 'TEamp', 56: 'STPi', 57: 'Tepd', 58: 'TEpv', 59: '1', 60: 'THTF', 61: '2930', 62: '23', 63: 'PBc', 64: 'AIP', 65: '7B', 66: 'FST', 67: '5', 68: '7op', 69: 'TEOm', 70: 'TEO', 71: 'STPc', 72: 'VIP', 73: 'V2', 74: 'LIP', 75: 'MST', 76: 'V4', 77: 'V3', 78: 'Pro.St.', 79: '31', 80: '7A', 81: '7m', 82: 'TPt', 83: 'MT', 84: 'V4t', 85: 'MIP', 86: 'V1', 87: 'PIP', 88: 'V3A', 89: 'V6A', 90: 'DP', 91: 'V6'}
            return label_mapping.get(voxel_value, "Unknown Brain Region")
        mha_file_path = animal_info.raw_3D_data_dir+'/91 areas parcellation fitted on M132 MRI/Coronal Parcellation.mha'
        mha_image = sitk.ReadImage(mha_file_path)
        mha_array = sitk.GetArrayFromImage(mha_image)
        spacing = mha_image.GetSpacing()
        for i in range(1, len(full_name)+1):

            brain_area = macaque_voxel_value_to_area_name(i)
            voxels = np.array(np.where(mha_array==i)).T
            area_voxel_dict[brain_area] = voxels * np.array([spacing[2], spacing[1], spacing[0]])
    if animal == 'human' and dataset_key == 'lariviere':
        def number_to_area_name(index, brain_type):

            atlas = pd.read_csv(animal_info.raw_3D_data_dir+'/Atlas.csv')
            if brain_type=='R':
                if index!=0:
                    return atlas.iloc[index-1]['ROI Name'][2:]
                else:
                    print('index=0 is not allowed')
                    return 'unknown'
            if brain_type=='L':
                if index!=0:
                    return atlas.iloc[index-1+180]['ROI Name'][2:]
                else:
                    print('index=0 is not allowed')
                    return 'unknown'
        area_name = pd.read_csv(animal_info.raw_3D_data_dir+'/Atlas.csv')
        img = nib.load(animal_info.raw_3D_data_dir+'/Q1-Q6_RelatedParcellation210.L.midthickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii')
        coord_L = img.darrays[0].data

        img = nib.load(animal_info.raw_3D_data_dir+'/cortex_ColeAnticevic_NetPartition_GlasserParcels_L.label.gii')
        labels = img.darrays[0].data

        unique_labels = np.unique(labels)

        for label in unique_labels:

            mask = (labels == label)
            points_with_label = coord_L[mask]

            area_voxel_dict[number_to_area_name(label, 'L')] = points_with_label
        area_voxel_dict.pop('unknown', None)
    cf.save_dict(area_voxel_dict, animal_info.threeD_data_dir + 'area_voxel_dict_3D')
    return area_voxel_dict


def generate_area_center_3D(animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    area_voxel_dict = get_area_voxel_3D(animal, dataset_key)
    area_center_dict = {}
    for area, voxels in area_voxel_dict.items():
        area_center_dict[area] = np.mean(voxels, axis=0)
    cf.save_dict(area_center_dict, animal_info.threeD_data_dir + 'area_center_dict_3D')
    return area_center_dict


def generate_name_location(animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)

    area_center_dict = generate_area_center(animal, dataset_key)
    name_location_dict = area_center_dict.copy()
    cf.save_dict(name_location_dict, animal_info.area_coords_dir + 'name_location_dict')
    return name_location_dict


def generate_angle_matrix(animal, dataset_key, dim='2D'):
    animal_info = AnimalInfo(animal, dataset_key)
    if dim == '2D':
        area_center_dict = animal_info.area_center_dict
        full_split_name = animal_info.full_split_name
        split_angle_matrix = cf.get_zero_df(full_split_name, full_split_name)
        for i, center_i in area_center_dict.items():
            for j, center_j in area_center_dict.items():
                split_angle_matrix.loc[i, j] = cf.get_angle(center_i, center_j, angle_range='-180:180')
        angle_matrix = convert_split_df_to_normal(split_angle_matrix, animal, dataset_key, method='mean')
        cf.save_pkl(angle_matrix, animal_info.angle_matrix_dir + 'angle_matrix_2D')
    else:
        area_center_dict_3D = animal_info.area_center_dict_3D
        full_name = animal_info.full_name
        azim_matrix = pd.DataFrame(np.zeros([len(full_name), len(full_name)]), index=full_name, columns=full_name)
        elev_matrix = pd.DataFrame(np.zeros([len(full_name), len(full_name)]), index=full_name, columns=full_name)
        for i, center_i in area_center_dict_3D.items():
            for j, center_j in area_center_dict_3D.items():
                azim_matrix.loc[i, j], elev_matrix.loc[i, j] = cf.get_angle_3d(center_i, center_j)
        angle_matrix = {'azim': azim_matrix, 'elev': elev_matrix}
        cf.save_pkl(angle_matrix, animal_info.angle_matrix_dir + 'angle_matrix_3D')
    return angle_matrix


def generate_distance_matrix(animal, dataset_key, dim='2D'):
    animal_info = AnimalInfo(animal, dataset_key)
    if dim == '2D':
        area_center_dict = animal_info.area_center_dict
        full_split_name = animal_info.full_split_name
        split_distance_matrix = cf.get_zero_df(full_split_name, full_split_name)
        for i, center_i in area_center_dict.items():
            for j, center_j in area_center_dict.items():
                split_distance_matrix.loc[i, j] = np.linalg.norm(center_i - center_j, ord=2)
        distance_matrix = convert_split_df_to_normal(split_distance_matrix, animal, dataset_key, method='min')
        cf.save_pkl(distance_matrix, animal_info.distance_matrix_dir + 'distance_matrix_2D')
    else:
        area_center_dict_3D = animal_info.area_center_dict_3D
        full_name = animal_info.full_name
        distance_matrix = pd.DataFrame(np.zeros([len(full_name), len(full_name)]), index=full_name, columns=full_name)
        for i, center_i in area_center_dict_3D.items():
            for j, center_j in area_center_dict_3D.items():
                distance_matrix.loc[i, j] = np.linalg.norm(center_i - center_j, ord=2)
        cf.save_pkl(distance_matrix, animal_info.distance_matrix_dir + 'distance_matrix_3D')
    return distance_matrix


def generate_clean_area_category(animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    area_category = cf.load_df(animal_info.raw_area_category_dir + 'area_category')
    clean_area_category = {}
    if animal_info.animal_and_dataset_key == 'mouse_gamanut':
        for area, row in area_category.iterrows():
            clean_area_category[area] = row['area_type']
        clean_area_category['PL'] = 'Prefrontal'
        clean_area_category['ORBl'] = 'Prefrontal'
        clean_area_category['ORBm'] = 'Prefrontal'
        clean_area_category['FRP'] = 'Prefrontal'
        clean_area_category['MOs'] = 'Prefrontal'
        clean_area_category['ILA'] = 'Prefrontal'
        clean_area_category['ACAd'] = 'Limbic'
        clean_area_category['ACAv'] = 'Limbic'
        clean_area_category['RSPv'] = 'Limbic'
        clean_area_category['RSPagl'] = 'Limbic'
        clean_area_category['RSPd'] = 'Limbic'
        clean_area_category['MM'] = 'Limbic'
    elif animal_info.animal_and_dataset_key == 'marmoset_majka':
        category_map = {
        'Visual': 'Occipital',
        'Insula and rostral lateral sulcus': 'Insular',
        'Lateral and inferior temporal': 'Temporal',
        'Orbitofrontal': 'Prefrontal',
        'Motor and premotor': 'Frontal',
        'Posterior cingulate, medial, and retrosplenial': 'Limbic',
        'Medial prefrontal': 'Prefrontal',
        'Auditory': 'Temporal',
        'Somatosensory': 'Parietal',
        'Ventral temporal': 'Temporal',
        'Posterior parietal': 'Parietal',
        'Ventrolateral prefrontal': 'Prefrontal',
        'Dorsolateral prefrontal': 'Prefrontal'
        }
        for area, row in area_category.iterrows():
            clean_area_category[area] = category_map[row['area_type']]
    elif animal_info.animal_and_dataset_key in ['macaque_markov29', 'macaque_markov40']:
        for area, row in area_category.iterrows():
            clean_area_category[area] = row['area_type']
        clean_area_category['INSULA'] = 'Insular'
        clean_area_category['Pi'] = 'Insular'
    elif animal_info.animal_and_dataset_key == 'human_lariviere':
        for area, row in area_category.iterrows():
            clean_area_category[area] = row['simple_area_type']
    else:
        raise NotImplementedError
    cf.save_dict(clean_area_category, animal_info.area_category_dir + 'area_category_dict')
    return clean_area_category


def generate_intrinsic_proportion(animal, datasetkey):
    animal_info = AnimalInfo(animal, datasetkey)
    intrinsic_proportion_dict = defaultdict(list)
    if animal_info.animal_and_dataset_key == 'marmoset_majka':
        SI_table_3 = cf.load_df(animal_info.raw_data_dir + 'SI_table_3')
        for _, row in SI_table_3.iterrows():
            area = row['Area']
            intrinsic_proportion_dict[area].append(row['Intrinsic'] / row['Total'])
    else:
        raise NotImplementedError
    intrinsic_proportion_dict = {k: np.mean(v) for k, v in intrinsic_proportion_dict.items()}
    cf.save_dict(intrinsic_proportion_dict, animal_info.intrinsic_proportion_dir + 'intrinsic_proportion_dict')
    return intrinsic_proportion_dict


def generate_injection_id_to_area_map(animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    if animal_info.animal_and_dataset_key == 'marmoset_majka':
        SI_table_3 = cf.load_df(animal_info.raw_data_dir + 'SI_table_3')
        injection_id_to_area_map = {}
        for _, row in SI_table_3.iterrows():

            processed_case = re.sub(r'[XRL]$', '', row['Case'])
            injection_id_to_area_map[cf.cat(processed_case, row['Tracer'], sep='-')] = row['Area']
        cf.save_dict(injection_id_to_area_map, animal_info.injection_to_area_map_dir + 'injection_id_to_area_map')
        return injection_id_to_area_map
    else:
        raise NotImplementedError


def generate_injection_site(animal, dataset_key, dim='2D'):
    animal_info = AnimalInfo(animal, dataset_key)

    injection_site = defaultdict(list)
    if animal_info.animal_and_dataset_key == 'marmoset_majka':
        if dim == '2D':
            df = cf.load_df(animal_info.raw_data_dir + '/flatmap_injection_location')
            for idx, row in df.iterrows():
                area = animal_info.injection_id_to_area_map[row['injection_id']]
                injection_site[area].append(np.array([row['flatmap_x'], row['flatmap_y']]))
        elif dim == '3D':
            SI_table_3 = cf.load_df(animal_info.raw_data_dir + '/SI_table_3')
            for index, row in SI_table_3.iterrows():
                area = row['Area']
                injection_site[area].append(np.array([row['Mediolateral'], row['Rostrocaudal'], row['Dorsoventral']]))
        cf.save_dict(injection_site, animal_info.injection_site_dir + f'injection_site_{dim}')
        return injection_site
    else:
        return NotImplementedError




class AnimalDir:
    def __init__(self, animal, dataset_key):
        self.animal = animal
        self.dataset_key = dataset_key
        self.animal_and_dataset_key = str(animal) + '_' + str(dataset_key)


        self.raw_data_dir = str(PROJECT_ROOT / 'raw_data' / self.animal_and_dataset_key) + os.sep
        self.raw_area_coords_dir = self.raw_data_dir + 'area_coords/'
        self.raw_whole_cortex_coords_dir = self.raw_data_dir + 'whole_cortex_coords/'
        self.raw_area_category_dir = self.raw_data_dir + 'area_category/'
        self.raw_3D_data_dir = self.raw_data_dir + 'threeD_data/'


        self.data_dir = str(PROJECT_ROOT / 'data' / self.animal_and_dataset_key) + os.sep
        self.area_coords_dir = self.data_dir + 'area_coords/'
        self.whole_cortex_coords_dir = self.data_dir + 'whole_cortex_coords/'
        self.area_category_dir = self.data_dir + 'area_category/'
        self.distance_matrix_dir = self.data_dir + 'distance_matrix/'
        self.angle_matrix_dir = self.data_dir + 'angle_matrix/'
        self.SC_dir = self.data_dir + 'SC/'
        self.intrinsic_proportion_dir = self.data_dir + 'intrinsic_proportion/'
        self.injection_to_area_map_dir = self.data_dir + 'injection_to_area_map/'
        self.injection_site_dir = self.data_dir + 'injection_site/'
        self.threeD_data_dir = self.data_dir + 'threeD_data/'


class AnimalInfo(AnimalDir):
    def __init__(self, animal, dataset_key):
        super().__init__(animal, dataset_key)
        self.force_re_preprocess = False

        self._re_preprocess_dict = {}

        SC_type_dict = {'mouse_gamanut': 'FLNe', 'marmoset_majka': 'FLNe', 'marmoset_V4': 'FLNe', 'macaque_markov29': 'FLNe', 'macaque_markov40': 'FLNe', 'human_lariviere': 'Dmri'}
        self.SC_type = SC_type_dict[self.animal_and_dataset_key]

        name_information = get_name_dict(self.animal, self.dataset_key)
        self.source_name = name_information['source_name']
        self.target_name = name_information['target_name']
        self.full_name = name_information['full_name']
        self.full_split_name = name_information['full_split_name']
        self.name_to_split_name_dict = name_information['name_to_split_name_dict']
        self.split_name_to_name_dict = name_information['split_name_to_name_dict']

        self.clustered_source_name, self.clustered_target_name, self.clustered_full_name = get_source_target_full_name_in_clustered_order(self.animal, self.dataset_key)

        self.column_mode = 'target'

    def set_force_re_preprocess(self):
        self.force_re_preprocess = True

    def _update_re_preprocess_dict(self, prop_name):
        self._re_preprocess_dict[prop_name] = True

    def _raise_for_force_re_preprocess(self, prop_name):
        if self.force_re_preprocess and not self._re_preprocess_dict.get(prop_name, False):

            raise ValueError('force_re_preprocess')

    @property
    def reordered_full_name(self):
        if self.SC_focus_direction == 'row':
            return prioritize_sublist(self.full_name, self.source_name)
        elif self.SC_focus_direction == 'col':
            return prioritize_sublist(self.full_name, self.target_name)

    @property
    def reordered_full_name_clustered(self):
        if self.SC_focus_direction == 'row':
            return prioritize_sublist(self.clustered_full_name, self.clustered_source_name)
        elif self.SC_focus_direction == 'col':
            return prioritize_sublist(self.clustered_full_name, self.clustered_target_name)

    @property
    def coords_dict(self):
        try:
            self._raise_for_force_re_preprocess('coords_dict')
            return load_coords_dict(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('coords_dict')
            return generate_coords_polygon(self.animal, self.dataset_key)['coords_dict']

    @property
    def polygon_dict(self):
        try:
            self._raise_for_force_re_preprocess('polygon_dict')
            return load_polygon_dict(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('polygon_dict')
            return generate_coords_polygon(self.animal, self.dataset_key)['polygon_dict']

    @property
    def whole_cortex_coords(self):
        try:
            self._raise_for_force_re_preprocess('whole_cortex_coords')
            return load_whole_cortex_coords(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('whole_cortex_coords')
            return generate_coords_polygon(self.animal, self.dataset_key)['whole_cortex_coords']

    @property
    def whole_cortex_polygon(self):
        try:
            self._raise_for_force_re_preprocess('whole_cortex_polygon')
            return load_whole_cortex_polygon(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('whole_cortex_polygon')
            return generate_coords_polygon(self.animal, self.dataset_key)['whole_cortex_polygon']

    @property
    def area_center_dict(self):
        try:
            self._raise_for_force_re_preprocess('area_center_dict')
            return load_area_center_dict(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('area_center_dict')
            return generate_area_center(self.animal, self.dataset_key)

    @property
    def area_center_dict_3D(self):
        try:
            self._raise_for_force_re_preprocess('area_center_dict_3D')
            return load_area_center_dict_3D(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('area_center_dict_3D')
            return generate_area_center_3D(self.animal, self.dataset_key)

    @property
    def name_location_dict(self):
        try:
            self._raise_for_force_re_preprocess('name_location_dict')
            return load_name_location_dict(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('name_location_dict')
            return generate_name_location(self.animal, self.dataset_key)

    @property
    def angle_2D(self):
        try:
            self._raise_for_force_re_preprocess('angle_2D')
            return load_angle_matrix(self.animal, self.dataset_key, dim='2D')
        except:
            self._update_re_preprocess_dict('angle_2D')
            return generate_angle_matrix(self.animal, self.dataset_key, dim='2D')

    @property
    def angle_3D(self):
        try:
            self._raise_for_force_re_preprocess('angle_3D')
            return load_angle_matrix(self.animal, self.dataset_key, dim='3D')
        except:
            self._update_re_preprocess_dict('angle_3D')
            return generate_angle_matrix(self.animal, self.dataset_key, dim='3D')

    @property
    def distance_2D(self):
        try:
            self._raise_for_force_re_preprocess('distance_2D')
            return load_distance_matrix(self.animal, self.dataset_key, dim='2D')
        except:
            self._update_re_preprocess_dict('distance_2D')
            return generate_distance_matrix(self.animal, self.dataset_key, dim='2D')

    @property
    def distance_3D(self):
        try:
            self._raise_for_force_re_preprocess('distance_3D')
            return load_distance_matrix(self.animal, self.dataset_key, dim='3D')
        except:
            self._update_re_preprocess_dict('distance_3D')
            return generate_distance_matrix(self.animal, self.dataset_key, dim='3D')

    @property
    def neuron_density(self):
        return load_neuron_density(self.animal, self.dataset_key)

    @property
    def connection_matrix_vmax(self):
        return 0.2

    @property
    def c(self):
        if self.animal == 'human':
            return 6
        elif self.animal == 'marmoset' and self.dataset_key == 'Dmri_Tian':
            raise NotImplementedError
        else:
            return 7

    @property
    def w(self):
        return 0.9

    @property
    def SC_focus_direction(self):
        row_animal_and_dataset_key_list = []
        col_animal_and_dataset_key_list = ['mouse_gamanut', 'marmoset_majka', 'marmoset_V4', 'macaque_markov29', 'macaque_markov40', 'human_lariviere']
        if self.animal_and_dataset_key in row_animal_and_dataset_key_list:
            return 'row'
        elif self.animal_and_dataset_key in col_animal_and_dataset_key_list:
            return 'col'

    @property
    def SC_data(self):
        SC = load_SC(self.animal, self.dataset_key)

        if self.animal_and_dataset_key in ['human_lariviere']:

            SC = normalize_SC(SC, animal=self.animal, dataset_key=self.dataset_key)


        SC = cf.reorder_df(SC, new_row=self.full_name, new_col=self.full_name)
        return SC

    @property
    def plot_intrinsic_proportion(self):
        try:
            self._raise_for_force_re_preprocess('plot_intrinsic_proportion')
            return load_intrinsic_proportion(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('plot_intrinsic_proportion')
            return generate_intrinsic_proportion(self.animal, self.dataset_key)

    @property
    def SCt_data(self):
        if self.animal_and_dataset_key == 'marmoset_majka':
            SC = self.SC_data.copy()
            for t in self.target_name:
                for s in self.source_name:
                    if s == t:
                        SC.loc[s, t] = self.plot_intrinsic_proportion[t]
                    else:
                        SC.loc[s, t] = SC.loc[s, t] * (1 - self.plot_intrinsic_proportion[t])
            return SC
        else:
            raise NotImplementedError

    @property
    def area_category(self):
        try:
            self._raise_for_force_re_preprocess('area_category')
            return load_area_category_dict(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('area_category')
            return generate_clean_area_category(self.animal, self.dataset_key)

    @property
    def injection_id_to_area_map(self):
        try:
            self._raise_for_force_re_preprocess('injection_id_to_area_map')
            return load_injection_id_to_area_map(self.animal, self.dataset_key)
        except:
            self._update_re_preprocess_dict('injection_id_to_area_map')
            return generate_injection_id_to_area_map(self.animal, self.dataset_key)

    @property
    def injection_site_2D(self):
        try:
            self._raise_for_force_re_preprocess('injection_site_2D')
            return load_injection_site(self.animal, self.dataset_key, dim='2D')
        except:
            self._update_re_preprocess_dict('injection_site_2D')
            return generate_injection_site(self.animal, self.dataset_key, dim='2D')

    @property
    def injection_site_3D(self):
        try:
            self._raise_for_force_re_preprocess('injection_site_3D')
            return load_injection_site(self.animal, self.dataset_key, dim='3D')
        except:
            self._update_re_preprocess_dict('injection_site_3D')
            return generate_injection_site(self.animal, self.dataset_key, dim='3D')


def load_coords_dict(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'coords_dict')


def load_polygon_dict(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'polygon_dict')


def load_whole_cortex_coords(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'whole_cortex_coords')


def load_whole_cortex_polygon(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'whole_cortex_polygon')


def load_area_center_dict(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'area_center_dict')


def load_area_center_dict_3D(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.threeD_data_dir + 'area_center_dict_3D')


def load_name_location_dict(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_coords_dir + 'name_location_dict')


def load_SC(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    SC = cf.load_df(animal_dir.SC_dir + 'SC_data')
    return SC


def load_intrinsic_proportion(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.intrinsic_proportion_dir + 'intrinsic_proportion_dict')


def load_angle_matrix(animal, dataset_key, dim='2D'):
    animal_dir = AnimalDir(animal, dataset_key)
    if dim == '2D':
        return cf.load_pkl(animal_dir.angle_matrix_dir + 'angle_matrix_2D')
    elif dim == '3D':
        return cf.load_pkl(animal_dir.angle_matrix_dir + 'angle_matrix_3D')


def load_distance_matrix(animal, dataset_key, dim='2D'):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.distance_matrix_dir + 'distance_matrix_' + dim)


@cf.to_be_improved
def load_neuron_density(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)


    animal_info = AnimalInfo(animal, dataset_key)
    return cf.create_dict(animal_info.full_name, 1)


def get_source_target_full_name(animal, dataset_key):
    animal_and_dataset_key = animal + '_' + dataset_key

    if animal_and_dataset_key == 'macaque_markov29':

        target_name = ['V1', 'V2', 'V4', 'DP', 'MT', '8m', '5', '8l', 'TEO', '2', 'F1', 'STPc', '7A', '46d', '10', '946v', '946d', 'F5', 'Tepd', 'PBr', '7m', '7B', 'F2', 'STPi', 'ProM', 'F7', '8B', 'STPr', '24c']
        source_name = ['10', '11', '12', '13', '14', '1', '23', '24a', '24b', '24c', '24d', '25', '2930', '2', '31', '32', '3', '44', '45A', '45B', '46d', '46v', '5', '7A', '7B', '7m', '7op', '8B', '8l', '8m', '8r', '946d', '946v', '9', 'AIP', 'CORE', 'DP', 'ENTORHINAL', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'FST', 'Gu', 'INSULA', 'IPa', 'LB', 'LIP', 'MB', 'MIP', 'MST', 'MT', 'OPAI', 'OPRO', 'PBc', 'PBr', 'PERIRHINAL', 'PGa', 'PIP', 'PIRIFORM', 'Pi', 'Pro.St.', 'ProM', 'SII', 'STPc', 'STPi', 'STPr', 'SUBICULUM', 'TEMPORAL_POLE', 'TEO', 'TEOm', 'TEad', 'TEama', 'TEamp', 'TEav', 'TEpv', 'THTF', 'TPt', 'Tepd', 'V1', 'V2', 'V3A', 'V3', 'V4', 'V4t', 'V6A', 'V6', 'VIP']

    if animal_and_dataset_key == 'macaque_markov40':
        target_name = ['V1', 'V2', 'V4', 'TEO', '946d', 'F5', '8m', '7A', 'DP', '2', '5', '7B', 'STPr', 'STPi', 'STPc', 'PBr', 'Tepd', '24c', 'F1', 'F2', 'F7', 'ProM', '8l', '946v', '46d', '8B', '10', 'MT', '7m', '9', '25', 'F3', 'F4', '32', 'LIP', 'OPRO', '3', '1', '45A', 'V6']
        source_name = ['10', '11', '12', '13', '14', '1', '23', '24a', '24b', '24c', '24d', '25', '2930', '2', '31', '32', '3', '44', '45A', '45B', '46d', '46v', '5', '7A', '7B', '7m', '7op', '8B', '8l', '8m', '8r', '946d', '946v', '9', 'AIP', 'CORE', 'DP', 'ENTORHINAL', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'FST', 'Gu', 'INSULA', 'IPa', 'LB', 'LIP', 'MB', 'MIP', 'MST', 'MT', 'OPAI', 'OPRO', 'PBc', 'PBr', 'PERIRHINAL', 'PGa', 'PIP', 'PIRIFORM', 'Pi', 'Pro.St.', 'ProM', 'SII', 'STPc', 'STPi', 'STPr', 'SUBICULUM', 'TEMPORAL_POLE', 'TEO', 'TEOm', 'TEad', 'TEama', 'TEamp', 'TEav', 'TEpv', 'THTF', 'TPt', 'Tepd', 'V1', 'V2', 'V3A', 'V3', 'V4', 'V4t', 'V6A', 'V6', 'VIP']

    if animal_and_dataset_key == 'marmoset_majka':
        target_name = ['A1-2', 'A10', 'A11', 'A19DI', 'A19M', 'A23a', 'A23b', 'A23c', 'A24d', 'A32', 'A32V', 'A3a', 'A3b', 'A45', 'A46D', 'A47L', 'A4ab', 'A4c', 'A6DC', 'A6DR', 'A6M', 'A6Va', 'A8C', 'A8aD', 'A8aV', 'A8b', 'A9', 'AIP', 'AuA1', 'AuCM', 'AuCPB', 'AuML', 'AuRT', 'LIP', 'MIP', 'MST', 'OPt', 'PE', 'PEC', 'PF', 'PFG', 'PG', 'PGM', 'PGa-IPa', 'S2E', 'TE3', 'TEO', 'TPO', 'V1', 'V2', 'V3A', 'V4', 'V4T', 'V5', 'V6']

        source_name = ['A1-2', 'A10', 'A11', 'A13L', 'A13M', 'A13a', 'A13b', 'A14C', 'A14R', 'A19DI', 'A19M', 'A23V', 'A23a', 'A23b', 'A23c', 'A24a', 'A24b', 'A24c', 'A24d', 'A25', 'A29a-c', 'A29d', 'A30', 'A31', 'A32', 'A32V', 'A35', 'A36', 'A3a', 'A3b', 'A45', 'A46D', 'A46V', 'A47L', 'A47M', 'A47O', 'A4ab', 'A4c', 'A6DC', 'A6DR', 'A6M', 'A6Va', 'A6Vb', 'A8C', 'A8aD', 'A8aV', 'A8b', 'A9', 'AI', 'AIP', 'APir', 'AuA1', 'AuAL', 'AuCL', 'AuCM', 'AuCPB', 'AuML', 'AuR', 'AuRM', 'AuRPB', 'AuRT', 'AuRTL', 'AuRTM', 'DI', 'Ent', 'FST', 'GI', 'Gu', 'IPro', 'LIP', 'MIP', 'MST', 'OPAl', 'OPro', 'OPt', 'PE', 'PEC', 'PF', 'PFG', 'PG', 'PGM', 'PGa-IPa', 'PaIL', 'PaIM', 'Pir', 'ProM', 'ProSt', 'ReI', 'S2E', 'S2I', 'S2PR', 'S2PV', 'STR', 'TE1', 'TE2', 'TE3', 'TEO', 'TF', 'TFO', 'TH', 'TL', 'TLO', 'TPO', 'TPPro', 'TPro', 'TPt', 'V1', 'V2', 'V3', 'V3A', 'V4', 'V4T', 'V5', 'V6', 'V6A', 'VIP']

    if animal_and_dataset_key == 'marmoset_riken':
        print('not tested')
        target_name = ['A10', 'A9', 'A8b', 'A46D', 'A46V', 'A8aD', 'A8aV', 'A47L', 'A47M', 'A45', 'A47O', 'ProM', 'A11', 'A13a', 'A13b', 'A13L', 'A13M', 'OPAl', 'OPro', 'Gu', 'A14C', 'A14R', 'A25', 'A32V', 'A32', 'A24a', 'A24b', 'A24c', 'A24d', 'A6DR', 'A6DC', 'A6Va', 'A6Vb', 'A6M', 'A8C', 'A4ab', 'A4c', 'AI', 'DI', 'GI', 'ReI', 'IPro', 'PaIL', 'PaIM', 'TPro', 'A3a', 'A3b', 'A1_2', 'S2E', 'S2I', 'S2PR', 'S2PV', 'AuRTL', 'AuRT', 'AuRPB', 'AuRTM', 'AuR', 'AuRM', 'AuAL', 'AuA1', 'AuCM', 'AuCPB', 'AuML', 'AuCL', 'STR', 'TPt', 'TE1', 'TE2', 'TE3', 'TEO', 'Pga_IPa', 'TPO', 'TPPro', 'Ent', 'A36', 'A35', 'TF', 'TL', 'TH', 'TLO', 'TFO', 'PaS', 'PrS', 'A23a', 'A23b', 'A23c', 'A23V', 'A29a_c', 'A29d', 'A30', 'ProSt', 'PGM', 'A31', 'PE', 'PEC', 'V6A', 'PF', 'PFG', 'OPt', 'PG', 'VIP', 'MIP', 'AIP', 'LIP', 'V1', 'V2', 'V3', 'V3A', 'A19DI', 'A19M', 'V4', 'V4T' 'V5', 'MST', 'FST', 'V6', 'Pir']
        source_name = ['A11', 'A45', 'A13M', 'A8b', 'A32', 'A24a', 'A8aD', 'A46V', 'A8aV', 'A46D', 'A6Vb', 'A24b', 'A6DR', 'A25', 'A14R', 'A13L', 'A6M', 'A9', 'A10', 'A47O']

    if animal_and_dataset_key == 'marmoset_V4':
        target_name = ['4', '9', '10', '12', '14', '15', '18', '19', '20', '21', '25', '27', '28', '29', '31', '33', '34', '35', '39', '40', '41', '45', '46', '48', '49', '50', '58', '62', '64', '66', '67', '72', '73', '74', '79', '81', '88', '91', '92', '94', '95']
        source_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96']

    if animal_and_dataset_key == 'mouse_gamanut':
        target_name = ['GU', 'SSp-un', 'MOp', 'VISC', 'SSs', 'PL', 'SSp-bfd', 'ACAd', 'RL', 'AL', 'DP', 'AUDpo', 'AM', 'MM', 'V1', 'LM', 'PM', 'P', 'RSPd']
        source_name = ['SSp-nm', 'SSp-lj', 'AIv', 'GU', 'AIp', 'FRP', 'SSp-ul', 'AId', 'SSp-un', 'SSp-ll', 'VISC', 'ORBl', 'PL', 'ORBm', 'ILA', 'MOp', 'MOs', 'SSs', 'ACAd', 'SSp-bfd', 'TEa', 'PERI', 'AUDv', 'AUDd', 'SSp-tr', 'ACAv', 'AUDp', 'ECT', 'A', 'RL', 'V1', 'RSPagl', 'TEp', 'RSPd', 'AM', 'RSPv', 'LM', 'MM', 'PM', 'AL', 'LI', 'POR', 'AUDpo', 'DP', 'LLA', 'PORa', 'P']

    if animal_and_dataset_key == 'human_lariviere':
        target_name = ['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', '4', '3b', 'FEF', 'PEF', '55b', 'V3A', 'RSC', 'POS2', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT', 'MT', 'A1', 'PSL', 'SFL', 'PCV', 'STV', '7Pm', '7m', 'POS1', '23d', 'v23ab', 'd23ab', '31pv', '5m', '5mv', '23c', '5L', '24dd', '24dv', '7AL', 'SCEF', '6ma', '7Am', '7PL', '7PC', 'LIPv', 'VIP', 'MIP', '1', '2', '3a', '6d', '6mp', '6v', 'p24pr', '33pr', 'a24pr', 'p32pr', 'a24', 'd32', '8BM', 'p32', '10r', '47m', '8Av', '8Ad', '9m', '8BL', '9p', '10d', '8C', '44', '45', '47l', 'a47r', '6r', 'IFJa', 'IFJp', 'IFSp', 'IFSa', 'p9-46v', '46', 'a9-46v', '9-46d', '9a', '10v', 'a10p', '10pp', '11l', '13l', 'OFC', '47s', 'LIPd', '6a', 'i6-8', 's6-8', '43', 'OP4', 'OP1', 'OP2-3', '52', 'RI', 'PFcm', 'PoI2', 'TA2', 'FOP4', 'MI', 'Pir', 'AVI', 'AAIC', 'FOP1', 'FOP3', 'FOP2', 'PFt', 'AIP', 'EC', 'PreS', 'H', 'ProS', 'PeEc', 'STGa', 'PBelt', 'A5', 'PHA1', 'PHA3', 'STSda', 'STSdp', 'STSvp', 'TGd', 'TE1a', 'TE1p', 'TE2a', 'TF', 'TE2p', 'PHT', 'PH', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP2', 'IP1', 'IP0', 'PFop', 'PF', 'PFm', 'PGi', 'PGs', 'V6A', 'VMV1', 'VMV3', 'PHA2', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', '31pd', '31a', 'VVC', '25', 's32', 'pOFC', 'PoI1', 'Ig', 'FOP5', 'p10p', 'p47r', 'TGv', 'MBelt', 'LBelt', 'A4', 'STSva', 'TE1m', 'PI', 'a32pr', 'p24']
        source_name = target_name.copy()

    if len(target_name) > len(source_name):
        full_name = target_name
    else:
        full_name = source_name
    return source_name, target_name, full_name


def get_source_target_full_name_in_clustered_order(animal, dataset_key):
    animal_and_dataset_key = animal + '_' + dataset_key
    if animal_and_dataset_key == 'mouse_gamanut':
        clustered_target_name = ['GU', 'SSp-un', 'MOp', 'VISC', 'SSs', 'PL', 'SSp-bfd', 'ACAd', 'RL', 'AL', 'DP', 'AUDpo', 'AM', 'MM', 'V1', 'LM', 'PM', 'P', 'RSPd']
        clustered_source_name = ['SSp-nm', 'SSp-lj', 'AIv', 'GU', 'AIp', 'FRP', 'SSp-ul', 'AId', 'SSp-un', 'SSp-ll', 'VISC', 'ORBl', 'PL', 'ORBm', 'ILA', 'MOp', 'MOs', 'SSs', 'ACAd', 'SSp-bfd', 'TEa', 'PERI', 'AUDv', 'AUDd', 'SSp-tr', 'ACAv', 'AUDp', 'ECT', 'A', 'RL', 'V1', 'RSPagl', 'TEp', 'RSPd', 'AM', 'RSPv', 'LM', 'MM', 'PM', 'AL', 'LI', 'POR', 'AUDpo', 'DP', 'LLA', 'PORa', 'P']

    if animal_and_dataset_key == 'marmoset_majka':
        clustered_source_name = ['AuA1', 'AuCPB', 'AuRT', 'AuCM', 'AuML', 'A32', 'A32V', 'A46D', 'A9', 'A45', 'A11', 'A8b', 'PGa-IPa', 'TE3', 'A10', 'TPO', 'PF', 'S2E', 'A6Va', 'A24d', 'A6M', 'A3b', 'A4c', 'PE', 'A1-2', 'A23c', 'A6DC', 'A3a', 'A4ab', 'MIP', 'PEC', 'TEO', 'V1', 'V2', 'V4', 'V4T', 'V3A', 'V6', 'A19DI', 'V5', 'A23b', 'A8aD', 'A8C', 'A6DR', 'A8aV', 'AIP', 'A47L', 'PG', 'A19M', 'PFG', 'MST', 'LIP', 'OPt', 'A23a','PGM', 'AuCL', 'AuRPB', 'STR', 'AuRTM', 'PaIL', 'AuRTL', 'TPro', 'AuR', 'AuAL', 'AuRM', 'APir', 'PaIM', 'TPPro', 'A13b', 'A14R', 'Pir', 'A13a', 'A14C', 'A25', 'A35', 'Ent', 'AI', 'IPro', 'A36', 'TE2', 'TE1', 'OPAl', 'OPro', 'A24a', 'A47O', 'Gu', 'S2PR', 'S2PV', 'A6Vb', 'ProM', 'A31', 'A24b','A24c', 'TPt', 'DI', 'GI', 'ReI', 'S2I', 'A23V', 'ProSt', 'A30', 'A29a-c', 'A29d', 'A46V', 'A13M', 'A13L', 'A47M', 'V6A', 'VIP', 'TFO', 'TF', 'TL', 'TH', 'TLO', 'FST', 'V3']
        clustered_target_name = ['AuA1', 'AuCPB', 'AuRT', 'AuCM', 'AuML', 'A32', 'A32V', 'A46D', 'A9', 'A45', 'A11', 'A8b', 'PGa-IPa', 'TE3', 'A10', 'TPO', 'PF', 'S2E', 'A6Va', 'A24d', 'A6M', 'A3b', 'A4c', 'PE', 'A1-2', 'A23c', 'A6DC', 'A3a', 'A4ab', 'MIP', 'PEC', 'TEO', 'V1', 'V2', 'V4', 'V4T', 'V3A', 'V6', 'A19DI', 'V5', 'A23b', 'A8aD', 'A8C', 'A6DR', 'A8aV', 'AIP', 'A47L', 'PG', 'A19M', 'PFG', 'MST', 'LIP', 'OPt', 'A23a','PGM']

    if animal_and_dataset_key == 'macaque_markov29':
        clustered_source_name = ['24c', '25', '32', '23', '24a', '24b', '24d', '2930', '31', 'F1', 'F2', 'F3', 'F4', 'F5', 'F7', 'ProM', 'F6', 'Gu', 'TEO', 'V1', 'V2', 'V4', 'V3', 'V3A', '1', '2', '3', '5', '7A', '7B', '7m', 'DP',  'LIP', 'V6', '7op', 'AIP', 'INSULA', 'MIP', 'Pi', 'PIP', 'SII', 'V6A', 'VIP', '10', '45A', '46d', '8B', '8l', '8m', '9', '946d', '946v', 'OPRO', '11', '12', '13', '14', '44', '45B', '46v', '8r', 'OPAI', 'MT', 'PBr', 'STPc', 'STPi', 'STPr', 'Tepd','CORE', 'ENTORHINAL', 'FST', 'IPa', 'LB', 'MB', 'MST', 'PBc', 'PERIRHINAL', 'PGa', 'PIRIFORM', 'TEMPORAL_POLE', 'Pro.St.', 'SUBICULUM', 'TEama', 'TEamp', 'TEad', 'TEav', 'TEOm', 'TEpv', 'THTF', 'TPt', 'V4t']
        clustered_target_name = ['24c', 'F1', 'F2', 'F5', 'F7', 'ProM', 'TEO', 'V1', 'V2', 'V4', '2', '5', '7A', '7B', '7m', 'DP', '10', '46d', '8B', '8l', '8m', '946d', '946v', 'MT', 'PBr', 'STPc', 'STPi', 'STPr', 'Tepd']

    if animal_and_dataset_key == 'macaque_markov40':
        clustered_source_name = ['24c', '25', '32', '23', '24a', '24b', '24d', '2930', '31', 'F1', 'F2', 'F3', 'F4', 'F5', 'F7', 'ProM', 'F6', 'Gu', 'TEO', 'V1', 'V2', 'V4', 'V3', 'V3A', '1', '2', '3', '5', '7A', '7B', '7m', 'DP',  'LIP', 'V6', '7op', 'AIP', 'INSULA', 'MIP', 'Pi', 'PIP', 'SII', 'V6A', 'VIP', '10', '45A', '46d', '8B', '8l', '8m', '9', '946d', '946v', 'OPRO', '11', '12', '13', '14', '44', '45B', '46v', '8r', 'OPAI', 'MT', 'PBr', 'STPc', 'STPi', 'STPr', 'Tepd','CORE', 'ENTORHINAL', 'FST', 'IPa', 'LB', 'MB', 'MST', 'PBc', 'PERIRHINAL', 'PGa', 'PIRIFORM', 'TEMPORAL_POLE', 'Pro.St.', 'SUBICULUM', 'TEama', 'TEamp', 'TEad', 'TEav', 'TEOm', 'TEpv', 'THTF', 'TPt', 'V4t']
        clustered_target_name = ['24c', '25', '32', 'F1', 'F2', 'F3', 'F4', 'F5', 'F7', 'ProM', 'TEO', 'V1', 'V2', 'V4', '1', '2', '3', '5', '7A', '7B', '7m', 'DP', 'LIP', 'V6', '10', '45A', '46d', '8B', '8l', '8m', '9', '946d', '946v', 'OPRO', 'MT', 'PBr', 'STPc', 'STPi', 'STPr', 'Tepd']

    if animal_and_dataset_key == 'human_lariviere':
        clustered_source_name = ['V1', 'MST', 'V6', 'V2', 'V3', 'V4', 'V8', '4', '3b', 'FEF', 'PEF', '55b', 'V3A', 'RSC', 'POS2', 'V7', 'IPS1', 'FFC', 'V3B', 'LO1', 'LO2', 'PIT', 'MT', 'A1', 'PSL', 'SFL', 'PCV', 'STV', '7Pm', '7m', 'POS1', '23d', 'v23ab', 'd23ab', '31pv', '5m', '5mv', '23c', '5L', '24dd', '24dv', '7AL', 'SCEF', '6ma', '7Am', '7PL', '7PC', 'LIPv', 'VIP', 'MIP', '1', '2', '3a', '6d', '6mp', '6v', 'p24pr', '33pr', 'a24pr', 'p32pr', 'a24', 'd32', '8BM', 'p32', '10r', '47m', '8Av', '8Ad', '9m', '8BL', '9p', '10d', '8C', '44', '45', '47l', 'a47r', '6r', 'IFJa', 'IFJp', 'IFSp', 'IFSa', 'p9-46v', '46', 'a9-46v', '9-46d', '9a', '10v', 'a10p', '10pp', '11l', '13l', 'OFC', '47s', 'LIPd', '6a', 'i6-8', 's6-8', '43', 'OP4', 'OP1', 'OP2-3', '52', 'RI', 'PFcm', 'PoI2', 'TA2', 'FOP4', 'MI', 'Pir', 'AVI', 'AAIC', 'FOP1', 'FOP3', 'FOP2', 'PFt', 'AIP', 'EC', 'PreS', 'H', 'ProS', 'PeEc', 'STGa', 'PBelt', 'A5', 'PHA1', 'PHA3', 'STSda', 'STSdp', 'STSvp', 'TGd', 'TE1a', 'TE1p', 'TE2a', 'TF', 'TE2p', 'PHT', 'PH', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'DVT', 'PGp', 'IP2', 'IP1', 'IP0', 'PFop', 'PF', 'PFm', 'PGi', 'PGs', 'V6A', 'VMV1', 'VMV3', 'PHA2', 'V4t', 'FST', 'V3CD', 'LO3', 'VMV2', '31pd', '31a', 'VVC', '25', 's32', 'pOFC', 'PoI1', 'Ig', 'FOP5', 'p10p', 'p47r', 'TGv', 'MBelt', 'LBelt', 'A4', 'STSva', 'TE1m', 'PI', 'a32pr', 'p24']
        clustered_target_name = clustered_source_name.copy()

    if animal_and_dataset_key == 'marmoset_V4':
        clustered_source_name = ['86', '96', '49', '90', '21', '30', '42', '18', '84', '65', '77', '62', '5', '2', '80', '6', '48', '7', '43', '15', '16', '20', '9', '70', '25', '82', '14', '8', '17', '28', '57', '59', '83', '37', '61', '45', '64', '39', '12', '92', '79', '10', '29', '58', '1', '40', '69', '75', '34', '94', '66', '91', '72', '23', '71', '88', '27', '44', '60', '31', '4', '47', '11', '38', '76', '33', '93', '19', '46', '56', '63', '89', '32', '36', '53', '3', '52', '35', '74', '67', '87', '73', '95', '24', '22', '68', '41', '51', '50', '54', '78', '81', '13', '26', '55', '85']
        clustered_target_name = ['48', '25', '58', '14', '20', '21', '9', '15', '67', '74', '62', '35', '73', '49', '81', '95', '19', '46', '33', '41', '50', '64', '79', '28', '39', '92', '12', '29', '10', '45', '31', '88', '66', '72', '27', '91', '4', '34', '94', '18', '40']

    if len(clustered_target_name) > len(clustered_source_name):
        clustered_full_name = clustered_target_name
    else:
        clustered_full_name = clustered_source_name
    return clustered_source_name, clustered_target_name, clustered_full_name


def get_full_split_name(name_to_split_name_dict):
    full_split_name = []
    for name, name_to_split_name in name_to_split_name_dict.items():
        full_split_name.extend(name_to_split_name)
    return full_split_name


def get_name_dict(animal, dataset_key):
    animal_and_dataset_key = animal + '_' + dataset_key
    source_name, target_name, full_name = get_source_target_full_name(animal, dataset_key)
    name_to_split_name_dict = {}
    split_name_to_name_dict = {}

    if animal_and_dataset_key == 'macaque_markov29':
        for name in full_name:
            if name in ['11']:
                name_to_split_name_dict[name] = ['11--1', '11--2']
            else:
                name_to_split_name_dict[name] = [name]

        full_split_name = get_full_split_name(name_to_split_name_dict)

        for name in full_split_name:
            if name in ['11--1', '11--2']:
                split_name_to_name_dict[name] = '11'
            else:
                split_name_to_name_dict[name] = name

    if animal_and_dataset_key == 'macaque_markov40':
        for name in full_name:
            if name in ['11']:
                name_to_split_name_dict[name] = ['11--1', '11--2']
            else:
                name_to_split_name_dict[name] = [name]

        full_split_name = get_full_split_name(name_to_split_name_dict)

        for name in full_split_name:
            if name in ['11--1', '11--2']:
                split_name_to_name_dict[name] = '11'
            else:
                split_name_to_name_dict[name] = name

    if animal_and_dataset_key == 'marmoset_majka':
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

        full_split_name = get_full_split_name(name_to_split_name_dict)

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
        for name in full_name:
            name_to_split_name_dict[name] = [name]

        full_split_name = get_full_split_name(name_to_split_name_dict)

        for name in full_split_name:
            split_name_to_name_dict[name] = name

    if animal_and_dataset_key == 'marmoset_V4':
        for name in full_name:
            name_to_split_name_dict[name] = [name]
            if name in ['11']:
                name_to_split_name_dict[name] = ['11--1', '11--2']
            if name in ['43']:
                name_to_split_name_dict[name] = ['43--1', '43--2']
            if name in ['77']:
                name_to_split_name_dict[name] = ['77--1', '77--2']

        full_split_name = get_full_split_name(name_to_split_name_dict)

        for name in full_split_name:
            split_name_to_name_dict[name] = name
            if name in ['11--1', '11--2']:
                split_name_to_name_dict[name] = '11'
            if name in ['43--1', '43--2']:
                split_name_to_name_dict[name] = '43'
            if name in ['77--1', '77--2']:
                split_name_to_name_dict[name] = '77'

    if animal_and_dataset_key == 'mouse_gamanut':
        for name in full_name:
            name_to_split_name_dict[name] = [name]

        full_split_name = get_full_split_name(name_to_split_name_dict)

        for name in full_split_name:
            split_name_to_name_dict[name] = name

    if animal_and_dataset_key == 'human_lariviere':
        for name in full_name:
            name_to_split_name_dict[name] = [name]
            if name in ['10pp']:
                name_to_split_name_dict[name] = ['10pp--1', '10pp--2']
            if name in ['23c']:
                name_to_split_name_dict[name] = ['23c--1', '23c--2']
            if name in ['23d']:
                name_to_split_name_dict[name] = ['23d--1', '23d--2']
            if name in ['5mv']:
                name_to_split_name_dict[name] = ['5mv--1', '5mv--2']
            if name in ['a47r']:
                name_to_split_name_dict[name] = ['a47r--1', 'a47r--2']
            if name in ['p47r']:
                name_to_split_name_dict[name] = ['p47r--1', 'p47r--2']
            if name in ['AAIC']:
                name_to_split_name_dict[name] = ['AAIC--1', 'AAIC--2']
            if name in ['Pir']:
                name_to_split_name_dict[name] = ['Pir--1', 'Pir--2']
            if name in ['OFC']:
                name_to_split_name_dict[name] = ['OFC--1', 'OFC--2']
            if name in ['pOFC']:
                name_to_split_name_dict[name] = ['pOFC--1', 'pOFC--2']
            if name in ['PoI1']:
                name_to_split_name_dict[name] = ['PoI1--1', 'PoI1--2']
            if name in ['PoI2']:
                name_to_split_name_dict[name] = ['PoI2--1', 'PoI2--2']
            if name in ['PreS']:
                name_to_split_name_dict[name] = ['PreS--1', 'PreS--2']
            if name in ['ProS']:
                name_to_split_name_dict[name] = ['ProS--1', 'ProS--2']
            if name in ['RSC']:
                name_to_split_name_dict[name] = ['RSC--1', 'RSC--2']

        full_split_name = get_full_split_name(name_to_split_name_dict)

        for name in full_split_name:
            split_name_to_name_dict[name] = name
            if name in ['10pp--1', '10pp--2']:
                split_name_to_name_dict[name] = '10pp'
            if name in ['23c--1', '23c--2']:
                split_name_to_name_dict[name] = '23c'
            if name in ['23d--1', '23d--2']:
                split_name_to_name_dict[name] = '23d'
            if name in ['5mv--1', '5mv--2']:
                split_name_to_name_dict[name] = '5mv'
            if name in ['p47r--1', 'p47r--2']:
                split_name_to_name_dict[name] = 'p47r'
            if name in ['a47r--1', 'a47r--2']:
                split_name_to_name_dict[name] = 'a47r'
            if name in ['AAIC--1', 'AAIC--2']:
                split_name_to_name_dict[name] = 'AAIC'
            if name in ['Pir--1', 'Pir--2']:
                split_name_to_name_dict[name] = 'Pir'
            if name in ['OFC--1', 'OFC--2']:
                split_name_to_name_dict[name] = 'OFC'
            if name in ['pOFC--1', 'pOFC--2']:
                split_name_to_name_dict[name] = 'pOFC'
            if name in ['PoI1--1', 'PoI1--2']:
                split_name_to_name_dict[name] = 'PoI1'
            if name in ['PoI2--1', 'PoI2--2']:
                split_name_to_name_dict[name] = 'PoI2'
            if name in ['PreS--1', 'PreS--2']:
                split_name_to_name_dict[name] = 'PreS'
            if name in ['ProS--1', 'ProS--2']:
                split_name_to_name_dict[name] = 'ProS'
            if name in ['RSC--1', 'RSC--2']:
                split_name_to_name_dict[name] = 'RSC'

    name_dict = {'target_name': target_name, 'source_name': source_name, 'full_name': full_name, 'full_split_name': full_split_name, 'name_to_split_name_dict': name_to_split_name_dict, 'split_name_to_name_dict': split_name_to_name_dict}
    return name_dict


def combine_name(input_name, animal, dataset_key):
    combined_name = input_name
    animal_and_dataset_key = animal + '_' + dataset_key
    if animal_and_dataset_key in ['macaque_markov29', 'macaque_markov40']:
        if input_name=='F5' or input_name=='44':
            combined_name = 'F5-44'
        if input_name=='8m' or input_name=='8l' or input_name=='8r':
            combined_name = '8m-8l-8r'
        if input_name=='STPr' or input_name=='STPi' or input_name=='STPc':
            combined_name = 'STPr-STPi-STPc'
        if input_name=='7B' or input_name=='7op':
            combined_name = '7B-7op'
        if input_name=='1' or input_name=='2':
            combined_name = '1-2'
        if input_name=='TEO' or input_name=='TEOm':
            combined_name = 'TEO-TEOm'
        if input_name=='946d' or input_name=='46d':
            combined_name = '946d-46d'
        if input_name=='Tepd' or input_name=='TEama' or input_name=='TEamp':
            combined_name = 'Tepd-TEama-TEamp'
        if input_name=='F3' or input_name=='F6':
            combined_name = 'F3-F6'
        if input_name=='45A' or input_name=='45B':
            combined_name = '45A-45B'

        if input_name=='946v' or input_name=='46v':
            combined_name = '946v-46v'
        if input_name=='PGa' or input_name=='IPa':
            combined_name = 'PGa-IPa'

        if input_name=='TEad' or input_name=='TEpv':
            combined_name = 'TEad-TEpv'
    elif animal_and_dataset_key in ['marmoset_majka']:
        if input_name=='PEC' or input_name=='PE':
            combined_name = 'PEC-PE'
        if input_name=='PG' or input_name=='PFG':
            combined_name = 'PG-PFG'
        if input_name=='A4ab' or input_name=='A4c':
            combined_name = 'A4ab-A4c'
        if input_name=='A6Va' or input_name=='A6Vb':
            combined_name = 'A6Va-A6Vb'
        if input_name=='A8aD' or input_name=='A8aV':
            combined_name = 'A8aD-A8aV'
        if input_name=='A19M' or input_name=='V6':
            combined_name = 'A19M-V6'
        if input_name=='A3a' or input_name=='A3b':
            combined_name = 'A3a-A3b'
        if input_name=='A32' or input_name=='A32V':
            combined_name = 'A32-A32V'

        if input_name=='AuRPB' or input_name=='STR':
            combined_name = 'AuRPB-STR'
        if input_name=='AuA1' or input_name=='AuR' or input_name=='AuRT':
            combined_name = 'AuA1-AuR-AuRT'
        if input_name=='AuRM' or input_name=='AuRTM' or input_name=='AuCM':
            combined_name = 'AuRM-AuRTM-AuCM'
        if input_name=='AuCL' or input_name=='AuML' or input_name=='AuRTL' or input_name=='AuAL':
            combined_name = 'AuCL-AuML-AuRTL-AuAL'
        if input_name=='A47L' or input_name=='A47M' or input_name=='A47O':
            combined_name = 'A47L-A47M-A47O'
        if input_name=='A23a' or input_name=='A23b' or input_name=='A23c' or input_name=='A23V':
            combined_name = 'A23a-A23b-A23c-A23V'
        if input_name=='S2I' or input_name=='S2E' or input_name=='S2PV' or input_name=='S2PR':
            combined_name = 'S2I-S2E-S2PV-S2PR'

        if input_name=='TFO' or input_name=='TLO' or input_name=='TH' or input_name=='TF' or input_name=='TL':
            combined_name = 'TFO-TLO-TH-TF-TL'
        if input_name=='A36' or input_name=='A35':
            combined_name = 'A36-A35'
        if input_name=='ReI' or input_name=='GI' or input_name=='DI' or input_name=='AI' or input_name=='IPro':
            combined_name = 'ReI-GI-DI-AI-IPro'
        if input_name=='TPro' or input_name=='PaIL' or input_name=='PaIM':
            combined_name = 'TPro-PaIL-PaIM'
        if input_name=='A29a-c' or input_name=='A29d' or input_name=='A30':
            combined_name = 'A29a-c-A29d-A30'
        if input_name=='A14R' or input_name=='A14C':
            combined_name = 'A14R-A14C'
        if input_name=='A13a' or input_name=='A13b' or input_name=='A13L' or input_name=='A13M':
            combined_name = 'A13a-A13b-A13L-A13M'
    return combined_name


def transform_combined_name(combined_name, source_animal, source_datasetkey, target_animal, target_dataset_key):
    combined_name_marmoset_majka_list = ['V1', 'V2', 'V4', 'V5', 'PEC-PE', 'Opt', 'PG-PFG', 'PGM', 'A4ab A4C', 'A6DC', 'A6DR', 'A6Va-A6Vb', 'A10', 'A8b', 'A8aD-A8aV', 'TPO', 'PF', 'A1-2', 'TEO', 'A46D', 'TE3', 'LIP', 'A19M V6', 'A3a-A3b', 'A8C', 'A6M', 'A9', 'A32-A32V', 'A45', 'AuRPB-STR', 'ProM', 'A46V', 'A24C', 'OPro', 'A25', 'AuA1-AuR-AuRT', 'AuCPB', 'AuRM-AuRTM-AuCM', 'AuCL-AuML-AuRTL-AuAL', 'A19DI', 'V4T', 'V3A', 'MST', 'MIP', 'A47L A47M A470', 'A23a A23b A23c A23V', 'PGa-IPa', 'A24d', 'AIP', 'S2I-S2E-S2PV-S2PR', 'A11', 'V3', 'FST', 'TE1', 'TE2', 'TPPro', 'TFO-TLO-TH-TF-TL', 'A36-A35', 'Ent', 'Pir', 'SUB', 'V6A', 'VIP', 'TPt', 'ReI-GI-DI-AI-IPro', 'TPro-PaIL-PaIM', 'A31', 'ProSt', 'A29a-c-A29d-A30', 'A24a', 'A24b', 'A14R-A14C', 'A13a-A13b-A13L-A13M', 'Gu', 'OPAl']
    combined_name_macaque_markov29_list = ['V1', 'V2', 'V4', 'MT', '5', 'DP', '7A', '7m', 'F1', 'F2', 'F7', 'F5-44', '10', '8B', '8m-8l-8r', 'STPr-STPi-STPc', '7B-7op', '1-2', 'TEO-TEOm', '946d-46d', 'Tepd-TEama-TEamp', 'LIP', 'V6', '3', 'F4', 'F3-F6', '9', '32', '45A-45B', 'PBr', 'ProM', '946v-46v', '24c', 'OPRO', '25', 'CORE', 'PBc', 'MB', 'LB', 'PIP', 'V4t', 'V3A', 'MST', 'MIP', '12', '23', 'PGa-IPa', '24d', 'AIP', 'SII', '11', 'V3', 'FST', 'TEav', 'TEad-TEpv', 'TEMPORAL_POLE', 'THTF', 'PIRIFORM', 'ENTORHINAL', 'PIR', 'SUBICULUM', 'V6A', 'VIP', 'TPt', 'INSULA', 'Pi', '31', 'Pro.St.', '2930', '24a', '24b', '14', '13', 'Gu', 'OPAI']
    combined_name_macaque_markov40_list = combined_name_macaque_markov29_list.copy()

    combined_name_dict = {'macaque_markov29': combined_name_macaque_markov29_list, 'macaque_markov40': combined_name_macaque_markov40_list, 'marmoset_majka': combined_name_marmoset_majka_list}
    combined_name_macaque_list = combined_name_dict[target_animal+'_'+target_dataset_key]

    source_animal_and_dataset_key = source_animal + '_' + source_datasetkey
    target_animal_and_dataset_key = target_animal + '_' + target_dataset_key
    idx = combined_name_dict[source_animal_and_dataset_key].index(combined_name)
    output_name = combined_name_dict[target_animal_and_dataset_key][idx]
    return output_name


def load_area_category_dict(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    return cf.load_pkl(animal_dir.area_category_dir + '/area_category_dict')


def get_area_category_in_order():


    return ['Frontal', 'Insular', 'Limbic', 'Occipital', 'Parietal', 'Temporal', 'Prefrontal']


def load_injection_id_to_area_map(animal, dataset_key):
    animal_dir = AnimalDir(animal, dataset_key)
    if animal_dir.animal_and_dataset_key == 'marmoset_majka':
        return cf.load_pkl(animal_dir.injection_id_to_area_map_dir + '/injection_id_to_area_map')
    else:
        raise ValueError('Only marmoset_majka has injection id to area map data.')


def load_injection_site(animal, dataset_key, dim='2D'):
    animal_dir = AnimalDir(animal, dataset_key)
    if animal_dir.animal_and_dataset_key == 'marmoset_majka':
        if dim == '2D':
            return cf.load_pkl(animal_dir.injection_site_dir + '/injection_site_2D')
        if dim == '3D':
            return cf.load_pkl(animal_dir.injection_site_dir + '/injection_site_3D')
    else:
        raise ValueError('Only marmoset_majka has injection site data.')


def get_molnar_performance(animal, dataset_key):
    macaque_markov29_dict = {
        'weak': {'MAE': 1.081, 'RMAE': 0.460},
        'weak_medium': {'MAE': 1.173, 'RMAE': 0.365},
        'medium_strong': {'MAE': 1.255, 'RMAE': 0.288},
        'strong': {'MAE': 1.324, 'RMAE': 0.237},
        'all': {'MAE': 1.207, 'RMAE': 0.336},
        'non': {'MAE': 1.498, 'RMAE': 1.101},
        'both': {'MAE': 1.318, 'RMAE': 0.628}
    }

    mouse_gamanut_dict = {
        'weak': {'MAE': 1.032, 'RMAE': 0.446},
        'weak_medium': {'MAE': 0.647, 'RMAE': 0.196},
        'medium_strong': {'MAE': 0.565, 'RMAE': 0.127},
        'strong': {'MAE': 0.569, 'RMAE': 0.102},
        'all': {'MAE': 0.622, 'RMAE': 0.166},
        'non': {'MAE': 2.911, 'RMAE': 2.288},
        'both': {'MAE': 0.683, 'RMAE': 0.222}
    }

    animal_and_dataset_key = animal + '_' + dataset_key

    if animal_and_dataset_key == 'macaque_markov29':
        return macaque_markov29_dict
    if animal_and_dataset_key == 'mouse_gamanut':
        return mouse_gamanut_dict




def load_SC_model_dict(outcomedir):
    return load_connectivity(outcomedir)




def whether_in_area(p, polygon):
    point = shapely.geometry.Point(p[0], p[1])
    return polygon.intersects(point)


def _get_poisson_points(hull, intensity=None, size=None, seed=None, rng=None):
    if rng is not None and seed is not None:
        raise ValueError("Cannot specify both rng and seed")
    if rng is None and seed is None:
        seed = cf.get_time_seed()
        rng = cf.get_local_rng(seed=seed)
    if rng is None and seed is not None:
        rng = cf.get_local_rng(seed=seed)
    if rng is not None and seed is None:
        pass

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


def get_random_points_in_polygon(polygon, density, seed):
    return _get_poisson_points(polygon, size=int(density*polygon.area), seed=seed)


def get_neuron_num(neuron_density, polygon_dict, total_num, animal, dataset_key):
    split_neuron_density = broadcast_value_to_split_area(neuron_density, animal, dataset_key)
    neuron_num = {}
    for area in split_neuron_density.keys():
        neuron_num[area] = split_neuron_density[area]*polygon_dict[area].area

    neuron_num_sum = sum(neuron_num.values())
    for area in neuron_num.keys():
        neuron_num[area] = int(neuron_num[area]/neuron_num_sum*total_num)
    return neuron_num


def point_in_which_area(points, polygon_dict, whole_cortex_polygon):
    indices = []
    for s in points:
        idx = 'out_of_areas'
        if whether_in_area(s, whole_cortex_polygon):
            for name in polygon_dict.keys():
                if whether_in_area(s, polygon_dict[name]):
                    idx = name
                    break
        indices.append(idx)
    return indices


def get_direction_method_type(direction_method):
    if direction_method in ['song', 'song_without_self', 'song_exclude_inside']:
        return 'force'
    if direction_method in ['random']:
        return 'random'


def get_connection_part_results(animal, dataset_key, direction_method, length_method, direction_params, length_params, na, point_density, part_index, process_num, the_seed, model_type='send', max_iteration=10):
    animal_info = AnimalInfo(animal, dataset_key)
    polygon_dict = animal_info.polygon_dict
    whole_cortex_polygon = animal_info.whole_cortex_polygon
    area_center_dict = animal_info.area_center_dict


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

    seed = the_seed * process_num + part_index
    rng = cf.get_local_rng(seed=seed)

    neuron_num = get_neuron_num(neuron_density=point_density, polygon_dict=polygon_dict, total_num=int(na), animal=animal, dataset_key=dataset_key)
    direction_method_type = get_direction_method_type(direction_method)

    for area, polygon in polygon_dict.items():
        multi_s = np.atleast_2d(_get_poisson_points(polygon, size=neuron_num[area], rng=rng))
        for s in multi_s:
            index_s = [area]
            index_t = ['out_of_areas']
            try_num = 0

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


                    if model_type == 'send':
                        forces_sum = -np.sum(np.array(forces), axis=0)
                    else:
                        forces_sum = np.sum(np.array(forces), axis=0)
                    direction_growth = forces_sum / np.linalg.norm(forces_sum, ord=2)
                else:
                    if direction_method == 'random':
                        angle = rng.rand(1) * 2 * np.pi
                        direction_growth = np.array([np.cos(angle), np.sin(angle)])
                        direction_growth = direction_growth.reshape((2,))

                if length_method == 'exponential':
                    length = rng.exponential(scale=split_length_params['Lambda'][area])
                if length_method == 'power_law':
                    length = rng.pareto(split_length_params['Lambda'][area])
                t = s + length * direction_growth


                index_t = point_in_which_area([t], polygon_dict, whole_cortex_polygon)

            if 'out_of_areas' not in index_t:
                tt.append(t)
                ss.append(s)
                indices_tt.append(index_t)
                indices_ss.append(index_s)


    if model_type=='receive':

        bb = ss
        ss = tt
        tt = bb
        indices_bb = indices_ss
        indices_ss = indices_tt
        indices_tt = indices_bb

    ss = np.array(ss)
    tt = np.array(tt)
    indices_ss = np.array(indices_ss).flatten()
    indices_tt = np.array(indices_tt).flatten()
    results = {'ss': ss, 'tt': tt, 'indices_ss': indices_ss, 'indices_tt': indices_tt}
    return results


def get_SC_model_dict_from_CCMe_model(CCMe_model, animal, dataset_key, outcomes_dir):

    CCMe_sym_model = pd.DataFrame((CCMe_model.values+CCMe_model.values.T)/2, index=CCMe_model.index, columns=CCMe_model.columns)


    SC_model_dict = {'CCMe': CCMe_model, 'CCMe_sym': CCMe_sym_model}


    new_entries = {}
    for k, v in SC_model_dict.items():
        normalized = normalize_SC(v, animal, dataset_key)
        new_entries[f"{k}_normalized"] = normalized
        new_entries[f"{k}_normalized_log"] = log_SC(normalized, animal, dataset_key)


    SC_model_dict.update(new_entries)


    SC_model_dict['FLNe'] = CCM_to_FLN(CCMe_model)


    save_connectivity(SC_model_dict, outcomes_dir)
    return SC_model_dict


def get_SC_model_dict_from_CCMt_model(CCMt_model, animal, dataset_key, outcomes_dir):

    CCMe_model = cf.replace_df_diagonal(CCMt_model, replace_value=np.nan)


    CCMe_sym_model = pd.DataFrame((CCMe_model.values+CCMe_model.values.T)/2, index=CCMe_model.index, columns=CCMe_model.columns)


    SC_model_dict = {'CCMe': CCMe_model, 'CCMe_sym': CCMe_sym_model, 'CCMt': CCMt_model}


    new_entries = {}
    for k, v in SC_model_dict.items():
        normalized = normalize_SC(v, animal, dataset_key)
        new_entries[f"{k}_normalized"] = normalized
        new_entries[f"{k}_normalized_log"] = log_SC(normalized, animal, dataset_key)


    SC_model_dict.update(new_entries)


    SC_model_dict['FLNe'] = CCM_to_FLN(CCMe_model)
    SC_model_dict['FLNt'] = CCM_to_FLN(CCMt_model)


    save_connectivity(SC_model_dict, outcomes_dir)
    return SC_model_dict


def get_connectivity_from_connection_samples(animal, dataset_key, outcomes_dir=None, timedir=None, indices_ss=None, indices_tt=None):
    if indices_ss is None or indices_tt is None:
        connection_model = ConnectionGenerationModel()
        connection_model.set_timedir(timedir)
        indices_ss = connection_model.get_simulation_value('indices_ss')
        indices_tt = connection_model.get_simulation_value('indices_tt')

    animal_info = AnimalInfo(animal, dataset_key)
    full_name = animal_info.full_name
    full_split_name = animal_info.full_split_name


    idx_ss = area_name_to_index_array(indices_ss, full_split_name)
    idx_tt = area_name_to_index_array(indices_tt, full_split_name)


    split_CCMt_array = connection_pair_to_connection_matrix(idx_ss, idx_tt, area_num=len(full_split_name))


    split_CCMt_model = pd.DataFrame(split_CCMt_array, columns=full_split_name, index=full_split_name)


    CCMt_model = convert_split_df_to_normal(split_CCMt_model, animal, dataset_key, method='sum')
    return get_SC_model_dict_from_CCMt_model(CCMt_model, animal, dataset_key, outcomes_dir)


def SC_type_to_SC_dict_key(SC_type):
    SC_dict_key = SC_type
    if SC_type == 'Dmri':
        SC_dict_key = 'CCMe_sym_normalized'
    if SC_type == 'FLNe':
        SC_dict_key = 'FLNe'
    return SC_dict_key


def SC_type_to_SC_dict_key_include_diagonal(SC_type):
    SC_dict_key = SC_type
    if SC_type == 'Dmri':
        SC_dict_key = 'CCMt_sym_normalized'
    if SC_type == 'FLNe':
        SC_dict_key = 'FLNt'
    return SC_dict_key


class ConnectionGenerationModel(ConnectionModel):
    model_name = 'connection_generation'

    def __init__(self):
        super().__init__()
        self.set_default_params()

    def set_default_params(self):
        self.process_num = 1
        self.model_type = 'send'
        self.use_neuron_density_to_generate = False
        self.max_iteration = 10

    @property
    def point_density(self):
        if self.use_neuron_density_to_generate:
            return self.animal_info.neuron_density.copy()
        else:
            return cf.create_dict(self.animal_info.full_name, default_value=1)

    def set_params(self, params):
        super().set_params(params)
        self.animal_info = AnimalInfo(self.animal, self.dataset_key)

    def get_unique_Lambda(self):
        Lambda = self.length_params['Lambda']
        if isinstance(Lambda, dict):
            if len(set(Lambda.values())) == 1:
                return list(Lambda.values())[0]
            else:
                raise ValueError('Lambda is not unique')
        else:
            return Lambda

    def run_detail(self):
        generation_kwargs = {'animal': self.animal, 'dataset_key': self.dataset_key, 'direction_method': self.direction_method, 'length_method': self.length_method, 'direction_params': self.direction_params, 'length_params': self.length_params, 'na': int(self.na/self.process_num), 'point_density': self.point_density, 'process_num': self.process_num, 'model_type': self.model_type, 'max_iteration': self.max_iteration, 'the_seed': self.the_seed}
        generate_for_list = list(range(self.process_num))
        worker_count = getattr(self, 'worker_count', self.process_num)
        connection_part_results = cf.multi_process_list_for(process_num=worker_count, func=get_connection_part_results, kwargs=generation_kwargs, for_list=generate_for_list, for_idx_name='part_index')
        self.simulation_results = {
            key: np.concatenate([result[key] for result in connection_part_results], axis=0)
            for key in ['ss', 'tt', 'indices_ss', 'indices_tt']
        }

    @property
    def SC_model_dict(self):
        try:
            return load_SC_model_dict(self.outcomes_dir)
        except FileNotFoundError:
            indices_ss = self.get_simulation_value('indices_ss')
            indices_tt = self.get_simulation_value('indices_tt')
            return get_connectivity_from_connection_samples(self.animal, self.dataset_key, outcomes_dir=self.outcomes_dir, timedir=self.timedir, indices_ss=indices_ss, indices_tt=indices_tt)

    def get_suitable_SC_model(self):
        SC_dict_key = SC_type_to_SC_dict_key(self.animal_info.SC_type)
        return self.SC_model_dict[SC_dict_key]

    def get_masked_SC_as_data(self, SC_dict_key):
        masked_SC = self.SC_model_dict[SC_dict_key].copy()

        for r in masked_SC.index:
            for c in masked_SC.columns:
                if r not in self.animal_info.source_name or c not in self.animal_info.target_name:
                    masked_SC.loc[r, c] = np.nan
        return masked_SC

    def get_suitable_SC_model_masked_as_data(self):
        return self.get_masked_SC_as_data(SC_type_to_SC_dict_key(self.animal_info.SC_type))

    def get_data_model_correlation(self, diagonal=False):
        if diagonal:
            raise NotImplementedError
        else:
            SC_model = self.get_suitable_SC_model_masked_as_data()
            SC_data = self.animal_info.SC_data
            return data_model_correlation(SC_model, SC_data, self.animal, self.dataset_key, diagonal=diagonal)

    def get_area_level_correlation(self, diagonal=False):
        if diagonal:
            raise NotImplementedError
        else:
            SC_model = self.get_suitable_SC_model_masked_as_data()
            SC_data = self.animal_info.SC_data
            return area_level_correlation(SC_model, SC_data, self.animal, self.dataset_key, diagonal=diagonal)

    def get_relative_error(self, diagonal=False):
        if diagonal:
            raise NotImplementedError
        else:
            SC_model = self.get_suitable_SC_model_masked_as_data()
            SC_data = self.animal_info.SC_data
            return data_model_relative_error(SC_model, SC_data, self.animal, self.dataset_key, diagonal=diagonal)


def get_connection_generation_results(config, output_dir, worker_count=None, force_run=False):
    params = dict(config)
    if worker_count is not None:
        params['worker_count'] = worker_count
    model = ConnectionGenerationModel()
    model.set_up(params=params, basedir=output_dir, force_run=force_run)
    model.run()
    model.SC_model_dict
    save_json(
        {'correlation': float(model.get_data_model_correlation())},
        Path(model.run_dir) / 'metrics.json',
    )
    return model


def area_EDR_3d_connection_generation(animal, dataset_key, Lambda):
    animal_info = AnimalInfo(animal, dataset_key)
    area_center_dict_3D = animal_info.area_center_dict_3D

    CCMt = animal_info.SC_data.copy()
    for source_area in area_center_dict_3D.keys():
        source_center = np.array(area_center_dict_3D[source_area])
        for target_area in area_center_dict_3D.keys():
            target_center = np.array(area_center_dict_3D[target_area])
            distance = np.linalg.norm(source_center - target_center)
            CCMt.loc[source_area, target_area] = np.exp(-distance / Lambda)
    return CCMt


class AreaEDR3DModel(ConnectionGenerationModel):
    model_name = 'area_edr_3d'

    def set_default_params(self):
        pass

    def _check_unique_Lambda(self):
        try:
            unique_Lambda = self.get_unique_Lambda()
            return unique_Lambda
        except:
            raise ValueError('Lambda is not unique, cannot run AreaEDR3DModel.')

    def run_detail(self):
        CCMt_model = area_EDR_3d_connection_generation(self.animal, self.dataset_key, self._check_unique_Lambda())
        self.simulation_results = {'CCMt_model': CCMt_model}

    @property
    def SC_model_dict(self):
        try:
            return load_SC_model_dict(self.outcomes_dir)
        except FileNotFoundError:
            return get_SC_model_dict_from_CCMt_model(self.get_simulation_value('CCMt_model'), self.animal, self.dataset_key, self.outcomes_dir)


class VoxelEDR3DModel(ConnectionGenerationModel):
    model_name = 'voxel_edr_3d'

    pass


def calculate_weighted_matching_index(weighted_adj, epsilon=1e-10):
    if not np.allclose(weighted_adj, weighted_adj.T):
        raise ValueError("")

    n = weighted_adj.shape[0]
    matching_matrix = np.zeros((n, n))


    mask = np.ones(n, dtype=bool)


    exists = (weighted_adj > 0)

    for i in range(n):
        w_i = weighted_adj[i]
        e_i = exists[i]

        for j in range(i+1, n):
            w_j = weighted_adj[j]
            e_j = exists[j]


            mask[:] = True
            mask[[i, j]] = False


            common = e_i & e_j & mask
            any_neighbor = (e_i | e_j) & mask


            num = (w_i[common] + w_j[common]).sum()
            den = (w_i[any_neighbor] + w_j[any_neighbor]).sum()

            if den > epsilon:
                val = num / den
                matching_matrix[i, j] = val
                matching_matrix[j, i] = val

    return matching_matrix


def calculate_binary_matching_index(adj, epsilon=1e-10):
    binary_adj = (adj > 0).astype(float)
    return calculate_weighted_matching_index(binary_adj, epsilon)


def initialize_adjacency_matrix(SC_data, init_connection_num):
    area_num = len(SC_data)

    binary_adj_pd = pd.DataFrame(
        np.zeros((area_num, area_num), dtype=float),
        index=SC_data.index,
        columns=SC_data.columns
    )


    SC_values = SC_data.copy()
    np.fill_diagonal(SC_values.values, 0)


    flat_indices = np.argsort(SC_values.values, axis=None)[::-1][:int(init_connection_num/2)]

    for idx in flat_indices:
        i, j = np.unravel_index(idx, SC_values.shape)

        binary_adj_pd.iloc[i, j] = 1
        binary_adj_pd.iloc[j, i] = 1
    return binary_adj_pd


def calculate_probability_matrix(binary_adj_pd, distance_3D, Lambda, gamma, shift=1e-6, set_existing_to_zero=True):

    distance_component = np.exp(-distance_3D.values / Lambda)


    matching_index_component = calculate_binary_matching_index(binary_adj_pd.values)


    prob_matrix = distance_component * ((matching_index_component + shift) ** gamma)

    if set_existing_to_zero:

        prob_matrix[binary_adj_pd.values > 0] = 0


    total_weight = prob_matrix.sum()
    if total_weight > 0:
        prob_matrix /= total_weight


    return pd.DataFrame(
        prob_matrix,
        index=binary_adj_pd.index,
        columns=binary_adj_pd.columns
    )


def add_new_connection(binary_adj_pd, prob_matrix, rng):
    area_num = binary_adj_pd.shape[0]
    flat_probs = prob_matrix.values.flatten()


    flat_probs /= flat_probs.sum()

    chosen_idx = rng.choice(flat_probs.size, p=flat_probs)
    row_idx, col_idx = np.unravel_index(chosen_idx, prob_matrix.shape)


    row_label = binary_adj_pd.index[row_idx]
    col_label = binary_adj_pd.columns[col_idx]


    binary_adj_pd.loc[row_label, col_label] = 1
    binary_adj_pd.loc[col_label, row_label] = 1
    return binary_adj_pd


def betzel_like_connection_generation(animal, dataset_key, Lambda, gamma, init_connection_num, seed, final_connection_density=None):
    rng = cf.get_local_rng(seed=seed)
    animal_info = AnimalInfo(animal, dataset_key)


    distance_3D = animal_info.distance_3D
    SC_data = animal_info.SC_data


    assert distance_3D.index.equals(SC_data.index) and distance_3D.columns.equals(SC_data.columns), "Distance matrix and SC data must have the same index and columns."


    if final_connection_density is None:
        final_connection_density = get_connection_density(SC_data, animal, dataset_key)
    area_num = len(animal_info.full_name)
    final_connection_num = int(area_num * (area_num - 1) / 2 * final_connection_density)


    binary_adj_pd = initialize_adjacency_matrix(SC_data, init_connection_num)


    for _ in tqdm(range(final_connection_num - init_connection_num)):
        prob_matrix = calculate_probability_matrix(
            binary_adj_pd, distance_3D, Lambda, gamma
        )
        binary_adj_pd = add_new_connection(binary_adj_pd, prob_matrix, rng)

    return binary_adj_pd


def chen_like_weighted_assignment(animal, dataset_key, Lambda, gamma, binary_adj_pd, shift=1e-6):
    animal_info = AnimalInfo(animal, dataset_key)


    distance_3D = animal_info.distance_3D
    SC_data = animal_info.SC_data


    assert distance_3D.index.equals(SC_data.index) and distance_3D.columns.equals(SC_data.columns), "Distance matrix and SC data must have the same index and columns."

    prob_matrix = calculate_probability_matrix(binary_adj_pd, distance_3D, Lambda, gamma, set_existing_to_zero=False, shift=shift)

    weighted_adj_pd = prob_matrix * binary_adj_pd

    return weighted_adj_pd


def adapted_chen_like_weighted_assignment(animal, dataset_key, Lambda, gamma, binary_adj_pd, true_binary_adj_pd, shift=1e-6):
    animal_info = AnimalInfo(animal, dataset_key)


    distance_3D = animal_info.distance_3D
    SC_data = animal_info.SC_data


    assert distance_3D.index.equals(SC_data.index) and distance_3D.columns.equals(SC_data.columns), "Distance matrix and SC data must have the same index and columns."

    prob_matrix = calculate_probability_matrix(binary_adj_pd, distance_3D, Lambda, gamma, set_existing_to_zero=False, shift=shift)

    weighted_adj_pd = prob_matrix * true_binary_adj_pd

    return weighted_adj_pd


def modify_weighted_adj_by_matching_index(weighted_adj_pd, epsilon=1e-10):
    matching_index = calculate_weighted_matching_index(weighted_adj_pd.values, epsilon=epsilon)
    modified_weighted_adj_pd = weighted_adj_pd.copy()
    modified_weighted_adj_pd = modified_weighted_adj_pd * pd.DataFrame(matching_index, index=weighted_adj_pd.index, columns=weighted_adj_pd.columns)
    return modified_weighted_adj_pd


class BetzelLikeModel(ConnectionGenerationModel):
    model_name = 'betzel_like'

    def set_default_params(self):
        pass

    def run_detail(self):
        binary_adj_pd = betzel_like_connection_generation(
            self.animal,
            self.dataset_key,
            self.Lambda,
            self.gamma,
            self.init_connection_num,
            self.the_seed
        )
        weighted_adj_pd = chen_like_weighted_assignment(
            self.animal,
            self.dataset_key,
            self.Lambda,
            self.gamma,
            binary_adj_pd
        )
        self.simulation_results = {
            'CCMe_model': weighted_adj_pd,
        }

    @property
    def SC_model_dict(self):
        try:
            return load_SC_model_dict(self.outcomes_dir)
        except FileNotFoundError:
            return get_SC_model_dict_from_CCMe_model(self.get_simulation_value('CCMe_model'), self.animal, self.dataset_key, self.outcomes_dir)

    def check_fig(self, binary_adj_pd, weighted_adj_pd):
        fig, axes = cf.gfa(ncols=2)

        ax = axes[0]
        cf.sns_heatmap(ax, binary_adj_pd)
        cf.set_ax(ax, title='Binary Connection Matrix')

        ax = axes[1]
        cf.sns_heatmap(ax, weighted_adj_pd)
        cf.set_ax(ax, title='Weighted Connection Matrix')

        cf.save_fig(fig, self.figs_dir+'/connection_matrices')


class AdaptedBetzelLikeModel(BetzelLikeModel):
    model_name = 'adapted_betzel_like'

    def run_detail(self):
        animal_info = AnimalInfo(self.animal, self.dataset_key)
        binary_adj_pd = (animal_info.SC_data.copy() > 0).astype(int)
        if self.binary_mode == 'union':
            sym_binary_adj_pd = binary_adj_pd | binary_adj_pd.T
        elif self.binary_mode == 'intersection':
            sym_binary_adj_pd = binary_adj_pd & binary_adj_pd.T

        weighted_adj_pd = adapted_chen_like_weighted_assignment(
            self.animal,
            self.dataset_key,
            self.Lambda,
            self.gamma,
            sym_binary_adj_pd,
            true_binary_adj_pd=binary_adj_pd,
            shift=1e-6
        )
        self.simulation_results = {
            'CCMe_model': weighted_adj_pd,
        }





def connection_pair_to_connection_matrix(source_idx, target_idx, area_num=None):

    if area_num is None:
        area_num = max(np.max(source_idx), np.max(target_idx)) + 1


    matrix = np.zeros((area_num, area_num), dtype=np.int32)


    np.add.at(matrix, (source_idx, target_idx), 1)

    return matrix


def area_name_to_index_array(area_name_list, ordered_area_name):

    area_to_index = {area: idx for idx, area in enumerate(ordered_area_name)}

    index_array = [area_to_index[area] for area in area_name_list]
    return index_array


def convert_split_df_to_normal(split_df, animal, dataset_key, method='sum'):
    animal_info = AnimalInfo(animal, dataset_key)
    full_name = animal_info.full_name
    full_split_name = animal_info.full_split_name
    split_name_to_name_dict = animal_info.split_name_to_name_dict

    value_dict = defaultdict(list)

    for row in full_split_name:
        for col in full_split_name:
            merge_row = split_name_to_name_dict[row]
            merge_col = split_name_to_name_dict[col]
            value = split_df.loc[row, col]
            value_dict[(merge_row, merge_col)].append(value)

    index, columns = full_name, full_name
    if method == 'sum':
        data = {k: np.nansum(v) for k, v in value_dict.items()}
    elif method == 'mean':
        data = {k: np.nanmean(v) for k, v in value_dict.items()}
    elif method == 'min':
        data = {k: np.nanmin(v) for k, v in value_dict.items()}
    else:
        raise ValueError(f"Unsupported method: {method}")

    df = pd.DataFrame(0, index=index, columns=columns, dtype=float)
    for (row, col), value in data.items():
        df.loc[row, col] = value

    return df


def broadcast_value_to_split_area(value_dict, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    split_value_dict = {}
    for area in animal_info.full_name:
        if area in value_dict:
            for split_area in animal_info.name_to_split_name_dict[area]:
                split_value_dict[split_area] = value_dict[area]
    return split_value_dict


def assign_value_to_split_area(split_value_dict, key, value, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    for split_area in animal_info.name_to_split_name_dict[key]:
        split_value_dict[split_area] = value
    return split_value_dict


def CCM_to_FLN(CCM):
    FLN = cf.normalize_df_with_sum(CCM, row_or_col='col')
    return FLN


def normalize_SC(SC, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)

    sub_SC = cf.get_sub_df(df=SC, row_list=animal_info.source_name, col_list=animal_info.target_name)
    SC_sum = sub_SC.sum().sum()
    if animal_info.SC_focus_direction == 'col':
        new_SC = SC / SC_sum * len(animal_info.target_name)
    elif animal_info.SC_focus_direction == 'row':
        new_SC = SC / SC_sum * len(animal_info.source_name)

    return new_SC


def scale_SC_to_int(SC, minimal_tol=None):
    SC_without_nan = cf.process_nan(SC, nan_policy='fill', fill_value=0.)
    minimal_value = np.nanmin(SC_without_nan[SC_without_nan > 0].values)
    if minimal_tol is not None:
        if minimal_value < minimal_tol:
            minimal_value = minimal_tol
    SC_int = (SC_without_nan / minimal_value).astype(int)
    return SC_int


def weight_matrix_to_topo_matrix(weight_matrix):
    if isinstance(weight_matrix, pd.DataFrame):

        topo_matrix = (weight_matrix.values > 0).astype(int)
        return pd.DataFrame(topo_matrix, index=weight_matrix.index, columns=weight_matrix.columns)

    elif isinstance(weight_matrix, np.ndarray):

        return (weight_matrix > 0).astype(int)

    else:
        raise ValueError(" pandas.DataFrame  numpy.ndarray ")


def prioritize_sublist(l, sub_l):
    back = [item for item in l if item not in sub_l]
    return sub_l + back


def rearrange_SC(SC, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    reordered_full_name = animal_info.reordered_full_name
    new_SC = cf.reorder_df(SC, reordered_full_name, reordered_full_name)
    return new_SC


def log_SC(SC, animal, dataset_key, base=10, apply_c=False, apply_w=False):
    animal_info = AnimalInfo(animal, dataset_key)
    c = animal_info.c
    w = animal_info.w

    SC_log = np.log(SC) / np.log(base)

    SC_log = SC_log.replace([np.inf, -np.inf], np.nan)
    if apply_c:
        SC_log = SC_log + c
    if apply_w:

        SC_log = SC_log.replace(np.nan, w)

        SC_log[SC_log < w] = w
    return SC_log


def coarse_grain_to_combined(value_dict, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)

    if animal_info.animal_and_dataset_key not in ['macaque_markov29', 'macaque_markov40', 'marmoset_majka']:
        raise NotImplementedError

    combined_value_dict = {}
    count_dict = {}
    for area, value in value_dict.items():
        combined_name = combine_name(area, animal, dataset_key)
        count_dict[combined_name] = count_dict.get(combined_name, 0) + 1
        combined_value_dict[combined_name] = combined_value_dict.get(combined_name, 0) + value
    for combined_name in combined_value_dict.keys():
        combined_value_dict[combined_name] = combined_value_dict[combined_name] / count_dict[combined_name]
    return combined_value_dict


def get_weight_category_mask(animal, dataset_key, diagonal=False):
    animal_info = AnimalInfo(animal, dataset_key)
    w = animal_info.w
    SC_data = animal_info.SC_data
    SC_data = cf.get_sub_df(df=SC_data, row_list=animal_info.source_name, col_list=animal_info.target_name)
    SC_data_logged = log_SC(SC_data, animal, dataset_key, apply_c=True, apply_w=True)

    category_mask_dict = {}
    category_mask_dict['weak'] = (SC_data_logged > w) & (SC_data_logged < 3)
    category_mask_dict['weak_medium'] = (SC_data_logged > w) & (SC_data_logged < 5)
    category_mask_dict['medium_strong'] = SC_data_logged > 3
    category_mask_dict['strong'] = SC_data_logged > 5
    category_mask_dict['all'] = SC_data_logged > w
    category_mask_dict['non'] = SC_data_logged <= w
    category_mask_dict['both'] = (SC_data_logged > w) | (SC_data_logged <= w)

    if diagonal:
        pass
    else:

        for key in category_mask_dict.keys():
            category_mask_dict[key] = cf.replace_df_diagonal(category_mask_dict[key], replace_value=False)
    return category_mask_dict


def get_intrinsic_proportion_from_SCt(SCt, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    plot_intrinsic_proportion = {}
    if animal_info.animal_and_dataset_key == 'marmoset_majka':
        col_sum = cf.col_sum_df(SCt)
        for area in animal_info.full_name:
            plot_intrinsic_proportion[area] = SCt.loc[area, area] / col_sum[area]
    else:
        raise NotImplementedError
    return plot_intrinsic_proportion


def get_area_level_property_repeated_by_SC(SC, property_df, animal, dataset_key, minimal_tol=None):
    SC_int = scale_SC_to_int(SC, minimal_tol=minimal_tol)
    animal_info = AnimalInfo(animal, dataset_key)
    property_dict = {}

    if animal_info.SC_focus_direction == 'col':
        for col in animal_info.target_name:
            sub_property_df = cf.get_sub_df(df=property_df, col_list=[col])
            sub_SC_int = cf.get_sub_df(df=SC_int, col_list=[col])
            property_dict[col] = cf.repeat_data(sub_property_df, sub_SC_int)
    if animal_info.SC_focus_direction == 'row':
        for row in animal_info.source_name:
            sub_property_df = cf.get_sub_df(df=property_df, row_list=[row])
            sub_SC_int = cf.get_sub_df(df=SC_int, row_list=[row])
            property_dict[row] = cf.repeat_data(sub_property_df, sub_SC_int)
    return property_dict




def get_connection_density(SC, animal, dataset_key, diagonal=False):
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    sub_SC = cf.get_sub_df(df=SC, row_list=source_name, col_list=target_name)
    if diagonal:
        total_connection_num = sub_SC.size
        existing_connection_num = np.sum(sub_SC.values > 0)
    else:
        total_connection_num = sub_SC.size - np.min([len(source_name), len(target_name)])
        existing_connection_num = np.sum(sub_SC.values > 0) - np.sum(np.diag(sub_SC.values) > 0)
    connection_density = existing_connection_num / total_connection_num
    return connection_density


def data_model_correlation(SC_data, SC_model, animal, dataset_key, diagonal=False):
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    sub_SC_data = cf.get_sub_df(df=SC_data, row_list=source_name, col_list=target_name)
    sub_SC_model = cf.get_sub_df(df=SC_model, row_list=source_name, col_list=target_name)
    corr = cf.get_pd_rectangle_corr(sub_SC_data, sub_SC_model, diagonal=diagonal)
    return corr


def data_model_relative_error(SC_data, SC_model, animal, dataset_key, diagonal=False):
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    sub_SC_data = cf.get_sub_df(df=SC_data, row_list=source_name, col_list=target_name)
    sub_SC_model = cf.get_sub_df(df=SC_model, row_list=source_name, col_list=target_name)
    sub_SC_data_logged = log_SC(sub_SC_data, animal, dataset_key, apply_c=True, apply_w=True)
    sub_SC_model_logged = log_SC(sub_SC_model, animal, dataset_key, apply_c=True, apply_w=True)
    relative_error = cf.get_zero_df(index=source_name, columns=target_name)
    for source in source_name:
        for target in target_name:
            relative_error.loc[source, target] = np.abs(sub_SC_data_logged.loc[source, target] - sub_SC_model_logged.loc[source, target]) / sub_SC_data_logged.loc[source, target]
            if diagonal == False:
                if source == target:
                    relative_error.loc[source, target] = np.nan
    return relative_error


def data_model_RMAE_masked(SC_data, SC_model, animal, dataset_key, mask, diagonal=False):
    relative_error = data_model_relative_error(SC_data, SC_model, animal, dataset_key, diagonal=diagonal)
    RMAE = np.nansum(relative_error.values * mask.values) / np.nansum(mask.values)
    return RMAE


def get_performance_dict(SC_data, SC_model, animal, dataset_key, diagonal=False):
    performance_dict = {}
    performance_dict['correlation'] = data_model_correlation(SC_data=SC_data, SC_model=SC_model, animal=animal, dataset_key=dataset_key, diagonal=diagonal)
    mask_dict = get_weight_category_mask(animal=animal, dataset_key=dataset_key, diagonal=diagonal)
    for k in ['all', 'non', 'both', 'strong', 'medium_strong', 'weak_medium', 'weak']:
        performance_dict[cf.cat('accuracy', k)] = 1 - data_model_RMAE_masked(SC_data=SC_data, SC_model=SC_model, animal=animal, dataset_key=dataset_key, mask=mask_dict[k], diagonal=diagonal)
    return performance_dict


def SC_proportion_in_weight_category(animal, dataset_key):
    weight_category_dict = get_weight_category_mask(animal, dataset_key)
    SC_proportion_dict = {}
    for key, mask in weight_category_dict.items():
        SC_proportion_dict[key] = np.sum(mask.values) / np.sum(weight_category_dict['both'].values)
    return SC_proportion_dict


def area_level_correlation(SC_data, SC_model, animal, dataset_key, diagonal=False):
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    SC_focus_direction = animal_info.SC_focus_direction
    corr_dict = {}
    if SC_focus_direction == 'col':
        for target in target_name:
            SC_data_target = cf.get_sub_df(df=SC_data, row_list=source_name, col_list=[target])
            SC_model_target = cf.get_sub_df(df=SC_model, row_list=source_name, col_list=[target])
            corr_dict[target] = cf.get_pd_rectangle_corr(SC_data_target, SC_model_target, diagonal=diagonal)
    elif SC_focus_direction == 'row':
        for source in source_name:
            SC_data_source = cf.get_sub_df(df=SC_data, row_list=[source], col_list=target_name)
            SC_model_source = cf.get_sub_df(df=SC_model, row_list=[source], col_list=target_name)
            corr_dict[source] = cf.get_pd_rectangle_corr(SC_data_source, SC_model_source, diagonal=diagonal)
    return corr_dict


def get_matrix_without_longrange_connections(connection_matrix, distance_matrix, threshold, replacement_value):
    """Return a deterministic matrix copy with entries beyond the threshold replaced."""
    local_matrix = connection_matrix.copy()
    local_matrix[distance_matrix > threshold] = replacement_value
    return local_matrix


def get_rm_longrange_correlation_change(SC_data, SC_model, distance_matrix, animal, dataset_key, removed_value, point_num=100, rm_mode='data'):
    max_distance = np.nanmax(distance_matrix.values)
    corr_list = []
    for threshold in np.linspace(0, max_distance, point_num):
        SC_data_threshold = get_matrix_without_longrange_connections(SC_data, distance_matrix, threshold, removed_value)
        SC_model_threshold = get_matrix_without_longrange_connections(SC_model, distance_matrix, threshold, removed_value)
        if rm_mode == 'data' or rm_mode == 'both':
            SC_data_to_use = SC_data_threshold
        else:
            SC_data_to_use = SC_data
        if rm_mode == 'model' or rm_mode == 'both':
            SC_model_to_use = SC_model_threshold
        else:
            SC_model_to_use = SC_model

        corr = data_model_correlation(SC_data_to_use, SC_model_to_use, animal, dataset_key, diagonal=False)
        corr_list.append(corr)
    return corr_list


def angle_ks_with_uniform(angle_list):
    def map_to_pi_range(angle):
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    ks_distance_list = []
    for rotation in np.linspace(0, 2*np.pi, 20):
        new_angle_list = map_to_pi_range(np.array(angle_list) + rotation)
        uniform = st.uniform(loc=np.min(new_angle_list), scale=np.max(new_angle_list)-np.min(new_angle_list))
        ks_distance = cf.get_ks(new_angle_list, uniform.cdf)
        ks_distance_list.append(ks_distance)
    return np.min(ks_distance_list)


def get_angle_distribution_from_SC_dispersion(animal, dataset_key, SC, scale_value):
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    name_to_split_name_dict = animal_info.name_to_split_name_dict
    polygon_dict = animal_info.polygon_dict
    area_size = {
        area: sum(polygon_dict[split_area].area for split_area in split_areas)
        for area, split_areas in name_to_split_name_dict.items()
    }
    point_cache = {}

    def get_cached_point(split_area, seed):
        key = (split_area, seed)
        if key not in point_cache:
            point_cache[key] = _get_poisson_points(polygon_dict[split_area], size=1, seed=seed)
        return point_cache[key]

    angle_dict = {}
    for target in target_name:
        angle_dict[target] = []
        for split_target in name_to_split_name_dict[target]:
            for source in source_name:
                for split_source in name_to_split_name_dict[source]:
                    sample_count = int(
                        SC.loc[source, target]
                        * scale_value
                        * polygon_dict[split_target].area
                        / area_size[target]
                        * polygon_dict[split_source].area
                        / area_size[source]
                    )
                    for i in range(sample_count):
                        t = get_cached_point(split_target, i)
                        s = get_cached_point(split_source, i)
                        angle_dict[target].append(cf.get_angle(source=[s[0], s[1]], target=[t[0], t[1]]))
    return angle_dict


def get_direction_preference(SC_data, animal, dataset_key, mode='uniform', SC_model=None):
    animal_info = AnimalInfo(animal, dataset_key)

    data_angle_dict = get_angle_distribution_from_SC_dispersion(animal, dataset_key, SC_data, scale_value=10000)
    if mode == 'uniform':
        direction_preference_dict = {area: angle_ks_with_uniform(data_angle_dict[area]) for area in data_angle_dict.keys()}
    elif mode == 'model':
        model_angle_dict = get_angle_distribution_from_SC_dispersion(animal, dataset_key, SC_model, scale_value=10000)
        direction_preference_dict = {area: cf.get_ks(data_angle_dict[area], model_angle_dict[area]) for area in data_angle_dict.keys()}
    return direction_preference_dict


def get_category_longrange_connection(SC_data, animal, dataset_key, longrange_threshold, dim='2D'):
    animal_info = AnimalInfo(animal, dataset_key)

    if dim == '2D':
        distance_matrix = animal_info.distance_2D
    elif dim == '3D':
        distance_matrix = animal_info.distance_3D

    longrange_SC_data = SC_data[distance_matrix > longrange_threshold]

    longrange_connection_sum = {}
    longrange_possible_sum = {}

    for source in animal_info.source_name:
        source_category = animal_info.area_category[source]
        for target in animal_info.target_name:
            target_category = animal_info.area_category[target]
            if source_category != target_category:
                longrange_possible_sum[source_category] = longrange_possible_sum.get(source_category, 0) + 1
                longrange_possible_sum[target_category] = longrange_possible_sum.get(target_category, 0) + 1


                if np.isnan(longrange_SC_data.loc[source, target]):
                    continue
                else:
                    longrange_connection_sum[source_category] = longrange_connection_sum.get(source_category, 0) + longrange_SC_data.loc[source, target]
                    longrange_connection_sum[target_category] = longrange_connection_sum.get(target_category, 0) + longrange_SC_data.loc[source, target]


    longrange_connection_proportion = {}
    for category in longrange_connection_sum.keys():
        longrange_connection_proportion[category] = longrange_connection_sum[category] / longrange_possible_sum[category]

    results = {
        'longrange_connection_sum': longrange_connection_sum,
        'longrange_possible_sum': longrange_possible_sum,
        'longrange_connection_proportion': longrange_connection_proportion
    }
    return results


def max_distance_in_polygon(vertices):
    distance_matrix = scipy.spatial.distance_matrix(vertices, vertices)
    return np.max(distance_matrix)


def get_injection_deviation_rate(animal, dataset_key, mode='2D'):
    animal_info = AnimalInfo(animal, dataset_key)
    injection_deviation_rate_dict = {}
    target_name = animal_info.target_name
    if animal_info.animal_and_dataset_key == 'marmoset_majka':
        def get_radius(dim):
            radius = defaultdict(float)
            if dim == '2D':
                for area in target_name:
                    for spilit_area in animal_info.name_to_split_name_dict[area]:
                        radius[area] += max_distance_in_polygon(animal_info.coords_dict[spilit_area])/2
            else:
                atlas = nib.load(animal_info.raw_3D_data_dir+'/atlas_segmentation_200um.nii.gz')
                area_name = pd.read_excel(animal_info.raw_3D_data_dir+'/atlas_labels.xlsx')
                atlas_label = atlas.get_fdata()
                affine = atlas.affine
                for label in np.unique(atlas_label):
                    if int(label)!=0:
                        voxel = np.array(np.where(atlas_label==label)).T
                        distances = pdist(indices_to_mm(voxel, affine))
                        radius[area_name.loc[(int(label))-1, 'Area']] = np.max(distances)/2
            return radius
        def indices_to_mm(repeated_indices, affine):
            return np.array([repeated_indices[:, 0]*(-affine[0, 0])-affine[0, 3], repeated_indices[:, 1]*(-affine[1, 1])-affine[1, 3], repeated_indices[:, 2]*affine[2, 2]+affine[2, 3]]).T
        if mode == '2D':
            area_center_dict = animal_info.area_center_dict
            injection_site = animal_info.injection_site_2D
            radius = get_radius(mode)
            for area in target_name:
                distance = []
                for inj in injection_site[area]:

                    split_distance = []
                    for spilit_area in animal_info.name_to_split_name_dict[area]:
                        split_distance.append(np.linalg.norm(area_center_dict[spilit_area]-inj))
                    distance.append(np.min(split_distance))
                distance = np.mean(distance)
                injection_deviation_rate_dict[area] = distance/radius[area]
        if mode == '3D':
            area_center_dict = animal_info.area_center_dict_3D
            injection_site = animal_info.injection_site_3D
            radius = get_radius(mode)
            for area in target_name:
                distance = []
                for inj in injection_site[area]:
                    distance.append(np.linalg.norm(area_center_dict[area]-inj))
                distance = np.mean(distance)
                injection_deviation_rate_dict[area] = np.min([distance/radius[area], 1.0])
        return injection_deviation_rate_dict
    else:
        raise NotImplementedError


def transform_combined_name_value_dict(combined_name_value_dict, source_animal, source_datasetkey, target_animal, target_dataset_key):
    new_combined_name_value_dict = {}
    for combined_name, value in combined_name_value_dict.items():
        new_combined_name = transform_combined_name(combined_name, source_animal, source_datasetkey, target_animal, target_dataset_key)
        new_combined_name_value_dict[new_combined_name] = value
    return new_combined_name_value_dict


def angle_distance_independence_test(angle, distance, angle_bin_num, distance_bin_num):

    prob_matrix, angle_bin_edge, distance_bin_edge, *_ = cf.get_hist_2d(angle, distance, angle_bin_num, distance_bin_num)


    angle_hist, *_ = cf.get_hist(angle, angle_bin_edge)
    distance_hist, *_ = cf.get_hist(distance, distance_bin_edge)


    independent_prob_matrix = np.outer(angle_hist, distance_hist)


    kl = cf.get_kl_divergence(prob_matrix.flatten(), independent_prob_matrix.flatten())
    jsd = cf.get_jsd(prob_matrix.flatten(), independent_prob_matrix.flatten())
    result = {'KL': kl, 'JSD': jsd}
    return result


def SC_symmetry_index():
    pass


def connection_proportion_inside_threshold(threshold_list, animal, dataset_key, SC_mode='binary', distance_mode='2D'):
    animal_info = AnimalInfo(animal, dataset_key)
    SC_data = animal_info.SC_data
    processesd_SC_data = SC_data.copy()
    processesd_SC_data = cf.get_sub_df(processesd_SC_data, animal_info.source_name, animal_info.target_name)
    if SC_mode == 'binary':
        processesd_SC_data = (processesd_SC_data > 0).astype(int)
    elif SC_mode == 'weight':
        pass
    if distance_mode == '2D':
        distance_matrix = animal_info.distance_2D
    elif distance_mode == '3D':
        distance_matrix = animal_info.distance_3D
    sum_SC = np.nansum(processesd_SC_data.values)
    proportion_list = []
    for threshold in threshold_list:
        masked_SC = processesd_SC_data.copy()
        masked_SC[distance_matrix > threshold] = 0
        proportion = np.nansum(masked_SC.values) / sum_SC
        proportion_list.append(proportion)
    return proportion_list


def multiple_injection_difference_ratio():
    pass




def plot_show_value_on_flatmap(ax, animal, dataset_key, value_dict, cmap, cbar_label='', target=None, source=None, show_area_name=False, text_fontsize=cf.FONT_SIZE/3, norm_mode='linear', vmin=None, vmax=None, norm_kwargs=None, edgecolor=cf.BLACK, cbar=True, cbar_position=None, fill=True, adjust_lim=True, polygon_kwargs=None, cbar_kwargs=None, set_xylim=True):
    animal_info = AnimalInfo(animal, dataset_key)
    name_location_dict = animal_info.name_location_dict
    coords_dict = animal_info.coords_dict

    split_value_dict = broadcast_value_to_split_area(value_dict, animal, dataset_key)
    mask = {area: False for area in split_value_dict.keys()}
    mask_color = cf.TRANSPARENT
    if target is not None:
        mask = assign_value_to_split_area(mask, key=target, value=True, animal=animal, dataset_key=dataset_key)
        mask_color = TARGET_COLOR
        mask_tick = 'target'
        cbar_kwargs = cf.update_dict(cbar_kwargs, {'mask_tick': mask_tick})
    if source is not None:
        mask = assign_value_to_split_area(mask, key=source, value=True, animal=animal, dataset_key=dataset_key)
        mask_color = SOURCE_COLOR
        mask_tick = 'source'
        cbar_kwargs = cf.update_dict(cbar_kwargs, {'mask_tick': mask_tick})
    cf.plt_polygon_heatmap(ax, xy_dict=coords_dict, value_dict=split_value_dict, mask=mask, mask_color=mask_color, cbar_label=cbar_label, cmap=cmap, norm_mode=norm_mode, vmin=vmin, vmax=vmax, norm_kwargs=norm_kwargs, edgecolor=edgecolor, cbar=cbar, cbar_position=cbar_position, fill=fill, adjust_lim=adjust_lim, polygon_kwargs=polygon_kwargs, cbar_kwargs=cbar_kwargs)

    if show_area_name:
        for name, location in name_location_dict.items():
            cf.add_text(ax, x=location[0], y=location[1], text=name, fontsize=text_fontsize, ha='center', va='center', transform='data')

    ax.axis('off')
    ax.axis('equal')
    if set_xylim:
        whole_cortex_coords = animal_info.whole_cortex_coords
        xlim = [np.min(whole_cortex_coords[:, 0]), np.max(whole_cortex_coords[:, 0])]
        ylim = [np.min(whole_cortex_coords[:, 1]), np.max(whole_cortex_coords[:, 1])]
        new_xlim, new_ylim = cf.get_xylim_by_aspect_ratio(xlim, ylim)
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)


def plot_show_area_centers_on_flatmap(ax, animal, dataset_key, color=cf.BLACK, s=80, mode='only_largest', zorder=None):
    animal_info = AnimalInfo(animal, dataset_key)
    area_center_dict = animal_info.area_center_dict

    if mode == 'only_largest':

        filtered_area_center_dict = {}
        for area in animal_info.full_name:
            max_split = None
            max_area = 0
            for spilit_area in animal_info.name_to_split_name_dict[area]:
                spilit_area_size = animal_info.polygon_dict[spilit_area].area
                if spilit_area_size > max_area:
                    max_area = spilit_area_size
                    max_split = spilit_area
            filtered_area_center_dict[max_split] = area_center_dict[max_split]
        for area, center in filtered_area_center_dict.items():
            if zorder is not None:
                cf.plt_scatter(ax, x=center[0], y=center[1], color=color, s=s, zorder=zorder)
            else:
                cf.plt_scatter(ax, x=center[0], y=center[1], color=color, s=s)
    else:
        for area, center in area_center_dict.items():
            if zorder is not None:
                cf.plt_scatter(ax, x=center[0], y=center[1], color=color, s=s, zorder=zorder)
            else:
                cf.plt_scatter(ax, x=center[0], y=center[1], color=color, s=s)

def plot_show_connection_on_flatmap(ax, animal, dataset_key, SC, show_area_name=False, text_fontsize=cf.FONT_SIZE/3, vmin=0.001, vmax=0.2, edgecolor=cf.BLACK, set_xylim=True, selected_target_area_list=None, top_edges=0.1, mode='global'):
    animal_info = AnimalInfo(animal, dataset_key)
    plot_show_area_boundary(ax, animal, dataset_key, show_area_name=show_area_name, text_fontsize=text_fontsize, edgecolor=edgecolor, set_xylim=set_xylim)

    new_max = cf.LINE_WIDTH * 1.5
    new_min = cf.LINE_WIDTH * 0.05


    connections = []
    area_center_dict = animal_info.area_center_dict

    for area in selected_target_area_list:
        for source_area in animal_info.source_name:
            weight = SC.loc[source_area, area]
            if weight > 0:
                connections.append((source_area, area, weight))


    if top_edges is not None:
        if mode == 'global':

            connections.sort(key=lambda x: x[2], reverse=True)

            if 0 < top_edges <= 1:
                n_edges = int(len(connections) * top_edges)
            else:
                n_edges = int(top_edges)


            n_edges = min(n_edges, len(connections))
            connections = connections[:n_edges]

        elif mode == 'per_target':

            filtered_connections = []

            for target_area in selected_target_area_list:

                target_connections = [conn for conn in connections if conn[1] == target_area]
                if not target_connections:
                    continue


                target_connections.sort(key=lambda x: x[2], reverse=True)


                if isinstance(top_edges, (int, float)):

                    if 0 < top_edges <= 1:
                        n_edges = int(len(target_connections) * top_edges)
                    else:
                        n_edges = int(top_edges)
                elif isinstance(top_edges, dict):

                    if target_area in top_edges:
                        edge_value = top_edges[target_area]
                        if 0 < edge_value <= 1:
                            n_edges = int(len(target_connections) * edge_value)
                        else:
                            n_edges = int(edge_value)
                    else:
                        n_edges = len(target_connections)
                elif isinstance(top_edges, list):

                    idx = selected_target_area_list.index(target_area)
                    if idx < len(top_edges):
                        edge_value = top_edges[idx]
                        if 0 < edge_value <= 1:
                            n_edges = int(len(target_connections) * edge_value)
                        else:
                            n_edges = int(edge_value)
                    else:
                        n_edges = len(target_connections)
                else:
                    n_edges = len(target_connections)


                n_edges = max(1, min(n_edges, len(target_connections)))
                filtered_connections.extend(target_connections[:n_edges])

            connections = filtered_connections


    for source_area, area, weight in connections:

        max_source_split = None
        max_source_area = 0
        for spilit_source_area in animal_info.name_to_split_name_dict[source_area]:
            spilit_source_area_size = animal_info.polygon_dict[spilit_source_area].area
            if spilit_source_area_size > max_source_area:
                max_source_area = spilit_source_area_size
                max_source_split = spilit_source_area


        max_area_split = None
        max_area_size = 0
        for spilit_area in animal_info.name_to_split_name_dict[area]:
            spilit_area_size = animal_info.polygon_dict[spilit_area].area
            if spilit_area_size > max_area_size:
                max_area_size = spilit_area_size
                max_area_split = spilit_area


        if max_source_split and max_area_split:
            start = area_center_dict[max_source_split]
            end = area_center_dict[max_area_split]

            if weight >= vmax:
                linewidth = new_max
            elif weight <= vmin:
                linewidth = 0.
            else:
                linewidth = cf.scale_to_new_range(data=weight, old_min=vmin, old_max=vmax, new_min=new_min, new_max=new_max)

            cf.plt_line(ax, x=[start[0], end[0]], y=[start[1], end[1]], color=cf.CRIMSON, linewidth=linewidth, alpha=0.7)
    plot_show_area_centers_on_flatmap(ax, animal, dataset_key, color=cf.BLACK, mode='only_largest', zorder=10000)


def plot_show_area_boundary(ax, animal, dataset_key, show_area_name=False, text_fontsize=cf.FONT_SIZE/3, edgecolor=cf.BLACK, set_xylim=True):
    animal_info = AnimalInfo(animal, dataset_key)
    value_dict = {area: 0 for area in animal_info.full_name}
    cmap = cf.get_cmap([cf.TRANSPARENT])
    plot_show_value_on_flatmap(ax, animal, dataset_key, value_dict=value_dict, show_area_name=show_area_name, text_fontsize=text_fontsize, edgecolor=edgecolor, fill=False, cbar=False, cmap=cmap, set_xylim=set_xylim)


def plot_show_whole_cortex_boundary(ax, animal, dataset_key, edgecolor=cf.BLACK, set_xylim=True):
    animal_info = AnimalInfo(animal, dataset_key)
    whole_cortex_coords = animal_info.whole_cortex_coords

    cf.plt_polygon(ax, whole_cortex_coords, color=edgecolor, fill=False)

    ax.axis('off')
    ax.axis('equal')
    if set_xylim:
        xlim = [np.min(whole_cortex_coords[:, 0]), np.max(whole_cortex_coords[:, 0])]
        ylim = [np.min(whole_cortex_coords[:, 1]), np.max(whole_cortex_coords[:, 1])]
        new_xlim, new_ylim = cf.get_xylim_by_aspect_ratio(xlim, ylim)
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)


def plot_draw_rectangle_for_connection_matrix(ax, animal, dataset_key):
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


def plot_draw_mask_for_unmeasured_block(ax, animal, dataset_key):
    animal_info = AnimalInfo(animal, dataset_key)
    target_name = animal_info.target_name
    source_name = animal_info.source_name
    target_num = len(target_name)
    source_num = len(source_name)
    color = cf.GRAY
    cmap = cf.get_cmap([color])
    unmeasured_matrix = np.full((source_num, source_num), np.nan)
    if target_num < source_num:
        for i in range(source_num):
            if i < target_num:
                unmeasured_matrix[:, i] = np.nan
            else:
                unmeasured_matrix[:, i] = 1
        cf.add_text(ax, x=(target_num+source_num)/2, y=source_num/2, text='Unmeasured', fontsize=cf.LABEL_SIZE, ha='center', va='center', transform='data')
    if target_num > source_num:
        for i in range(target_num):
            if i < source_num:
                unmeasured_matrix[i, :] = np.nan
            else:
                unmeasured_matrix[i, :] = 1
        cf.add_text(ax, x=target_num/2, y=(target_num+source_num)/2, text='Unmeasured', fontsize=cf.LABEL_SIZE, ha='center', va='center', transform='data')
    cf.sns_heatmap(ax, unmeasured_matrix, show_xtick=False, show_ytick=False, cmap=cmap, cbar=False, heatmap_kwargs={'rasterized': True})


def plot_visualize_connection_matrix_heatmap(ax_data, ax_model, matrix_data, matrix_model, animal, dataset_key, log_matrix=False, topo=False, show_correlation=True):
    animal_info = AnimalInfo(animal, dataset_key)
    reordered_full_name_clustered = animal_info.reordered_full_name_clustered
    SC_type = animal_info.SC_type

    if log_matrix:
        c = animal_info.c
        w = animal_info.w
        local_matrix_data = log_SC(SC=matrix_data, animal=animal, SC_type=SC_type, apply_c=False, apply_w=False)
        local_matrix_model = log_SC(SC=matrix_model, animal=animal, SC_type=SC_type, apply_c=False, apply_w=False)
        vmin = w
        vmax = c
        add_geq = False

    else:
        local_matrix_data = matrix_data
        local_matrix_model = matrix_model
        vmin = 0.
        vmax = animal_info.connection_matrix_vmax
        add_geq = True

    if topo:
        cmap = plt.cm.hot
        local_matrix_data = weight_matrix_to_topo_matrix(local_matrix_data)
        local_matrix_model = weight_matrix_to_topo_matrix(local_matrix_model)
        vmin = 0.
        vmax = 1.
    else:
        cmap = plt.cm.jet

    local_matrix_data = cf.reorder_df(local_matrix_data, reordered_full_name_clustered, reordered_full_name_clustered)
    local_matrix_model = cf.reorder_df(local_matrix_model, reordered_full_name_clustered, reordered_full_name_clustered)


    local_matrix_data = cf.process_nan(local_matrix_data, nan_policy='fill', fill_value=0.)
    local_matrix_model = cf.process_nan(local_matrix_model, nan_policy='fill', fill_value=0.)

    ticks = [vmin, vmax]


    cbar_kwargs = {'add_geq': add_geq, 'ticks': ticks}
    cf.sns_heatmap(ax_data, local_matrix_data, show_xtick=False, show_ytick=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs, heatmap_kwargs={'rasterized': True})
    cf.sns_heatmap(ax_model, local_matrix_model, show_xtick=False, show_ytick=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kwargs=cbar_kwargs, heatmap_kwargs={'rasterized': True})


    plot_draw_mask_for_unmeasured_block(ax_data, animal=animal, dataset_key=dataset_key)


    plot_draw_rectangle_for_connection_matrix(ax_data, animal=animal, dataset_key=dataset_key)
    plot_draw_rectangle_for_connection_matrix(ax_model, animal=animal, dataset_key=dataset_key)

    cf.set_ax(ax_data, title='Data', xlabel='Target', ylabel='Source')
    cf.set_ax(ax_model, title='Model', xlabel='Target', ylabel='Source')

    if show_correlation:
        corr = data_model_correlation(SC_data=local_matrix_data, SC_model=local_matrix_model, animal=animal, dataset_key=dataset_key, diagonal=False)
        cf.add_text(ax_model, x=0.5, y=1.25, text='Corr='+str(round(corr, 3)), va='center', ha='center', fontsize=cf.LABEL_SIZE)


def plot_visualize_distribution_repeated_by_SC(ax, some_property, SC, minimal_tol=None, distribution_type='CDF', distribution_kwargs=None):
    SC_int = scale_SC_to_int(SC, minimal_tol=minimal_tol)
    repeated_property = cf.repeat_data(some_property, SC_int)
    distribution_kwargs = cf.update_dict({}, distribution_kwargs)
    if distribution_type == 'CDF':
        cf.plt_cdf(ax, repeated_property, **distribution_kwargs)
    elif distribution_type == 'hist':
        plt_hist_instance = cf.PltHist(ax, repeated_property, **distribution_kwargs)
        return plt_hist_instance


def plot_visualize_data_model_distribution_repeated_by_SC(ax, some_property, SC_data, SC_model, minimal_tol=None, distribution_type='CDF', distribution_kwargs=None):
    SC_data_int = scale_SC_to_int(SC_data, minimal_tol=minimal_tol)
    SC_model_int = scale_SC_to_int(SC_model, minimal_tol=minimal_tol)
    repeated_property_data = cf.repeat_data(some_property, SC_data_int)
    repeated_property_model = cf.repeat_data(some_property, SC_model_int)
    distribution_kwargs = cf.update_dict({}, distribution_kwargs)
    if distribution_type == 'CDF':
        cf.plt_compare_cdf_ks(ax, data1=repeated_property_data, data2=repeated_property_model, color1=DATA_COLOR, color2=MODEL_COLOR, label1='data', label2='model', **distribution_kwargs)
    elif distribution_type == 'hist':
        pass


def plot_visualize_data_uniform_distribution_repeated_by_SC(ax, some_property, SC_data, seed, minimal_tol=None, distribution_type='CDF', distribution_kwargs=None):
    SC_data_int = scale_SC_to_int(SC_data, minimal_tol=minimal_tol)
    repeated_property_data = cf.repeat_data(some_property, SC_data_int)
    rng = cf.get_local_rng(seed)
    uniform_data = rng.uniform(low=np.min(repeated_property_data), high=np.max(repeated_property_data), size=len(repeated_property_data))
    distribution_kwargs = cf.update_dict({}, distribution_kwargs)
    if distribution_type == 'CDF':
        cf.plt_compare_cdf_ks(ax, data1=uniform_data, data2=repeated_property_data, color1=cf.BLACK, color2=DATA_COLOR, label1='uniform', label2='data', fontsize=cf.TITLE_SIZE, cdf_kwargs={'linewidth': 1.5*cf.LINE_WIDTH}, **distribution_kwargs)
    elif distribution_type == 'hist':
        pass


def plot_visualize_macaque_marmoset_performance_similarity(ax, macaque_performance_dict, marmoset_performance_dict):
    combined_macaque_performance_dict = coarse_grain_to_combined(macaque_performance_dict, 'macaque', 'markov40')
    combined_marmoset_performance_dict = coarse_grain_to_combined(marmoset_performance_dict, 'marmoset', 'majka')

    transformed_combined_macaque_performance_dict = transform_combined_name_value_dict(combined_macaque_performance_dict, source_animal='macaque', source_datasetkey='markov40', target_animal='marmoset', target_dataset_key='majka')


    common_area = set(transformed_combined_macaque_performance_dict.keys()) & set(combined_marmoset_performance_dict.keys())
    insect_transformed_combined_macaque_performance_dict = cf.get_sub_dict(transformed_combined_macaque_performance_dict, common_area)
    insect_combined_marmoset_performance_dict = cf.get_sub_dict(combined_marmoset_performance_dict, common_area)

    x = list(insect_transformed_combined_macaque_performance_dict.values())
    y = list(insect_combined_marmoset_performance_dict.values())
    cf.plt_linregress(ax, x, y, show_list=['p', 'r'], scatter_kwargs={'s': 70.})

    right_move_x_dict = insect_combined_marmoset_performance_dict.copy()
    for key in right_move_x_dict.keys():
        right_move_x_dict[key] += 0.02

    text_obj_list = cf.add_multi_text_by_dict(ax, insect_transformed_combined_macaque_performance_dict, insect_combined_marmoset_performance_dict, transform='data', fontsize=cf.FONT_SIZE, ha='left', va='bottom')
    cf.auto_adjust_text(ax, text_obj_list, add_arrow=False, seed=0)

    cf.set_ax(ax, xlabel='Performance (macaque)', ylabel='Performance (marmoset)')
