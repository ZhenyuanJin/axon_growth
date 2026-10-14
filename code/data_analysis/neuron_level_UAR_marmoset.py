





import os
from collections import defaultdict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf


def get_weighted_centroid(data, coordinates):

    weighted_coordinates = coordinates.T * data

    sum_weighted_coordinates = np.sum(weighted_coordinates, axis=1)

    total_weight = np.sum(data)

    centroid = sum_weighted_coordinates / total_weight
    return centroid





xscale = 40.909798
yscale = 35.332






file_path_cells = r'../../raw_data/marmoset_V4/injection/all_cells.func.gii'
file_path_inject = r'../../raw_data/marmoset_V4/injection/all_inject.func.gii'


all_cells = nib.load(file_path_cells)
all_inject = nib.load(file_path_inject)
inject_num = len(all_inject.darrays)


print("Cells Data Num:", len(all_cells.darrays))
print("Inject Data Num:", len(all_inject.darrays))


print(np.max(all_cells.darrays[0].data))
print(np.max(all_inject.darrays[0].data))


flatmap_gii = nib.load('../../raw_data/marmoset_V4/Template/lh.full.flat.patch.surf.gii')
print(flatmap_gii)
print(flatmap_gii.meta.metadata)
print(flatmap_gii.darrays[0].data.shape)
print(flatmap_gii.darrays[1].data.shape)

flatmap = flatmap_gii.darrays[0].data


flatmap = np.array([flatmap[:, 1], -flatmap[:, 0], flatmap[:, 2]]).T


surface_gii = nib.load('../../raw_data/marmoset_V4/Template/lh.graymid.surf.gii')
surface = surface_gii.darrays[0].data


v4_label_gii = nib.load('../../raw_data/marmoset_V4/MBMv4-only/surfFS.lh.MBMv4_cortex_parcellation.label.gii')
print(v4_label_gii.darrays[0].data.shape)
print(set(v4_label_gii.darrays[0].data))
v4_label = v4_label_gii.darrays[0].data

paxinos_label_gii = nib.load('../../raw_data/marmoset_V4/Template/lh.MBM_cortex_vPaxinos.label.gii')
paxinos_label = paxinos_label_gii.darrays[0].data

paxinos_atlas_labels = pd.read_excel('../../raw_data/marmoset_V4/paxinos/atlas_labels.xlsx')
paxinos_area_name = {}
for i in range(len(paxinos_atlas_labels)):
    paxinos_area_name[i] = paxinos_atlas_labels['Area'][i]


fig, ax = plt.subplots()
plt.scatter(flatmap[:, 0], flatmap[:, 1], c=v4_label, s=1, cmap=plt.cm.get_cmap('Blues'))
xrange = np.max(flatmap[:, 0])-np.min(flatmap[:, 0])
yrange = np.max(flatmap[:, 1])-np.min(flatmap[:, 1])
print('xrange:', xrange)
print('yrange:', yrange)



full_areas = range(1, len(set(v4_label)))


distance = []
angle = []

inject_areas = []
injection_site = {}
max_site = {}
for i in range(len(all_cells.darrays)):
    tracer_result = {}
    print('-------------------')
    print('i:', i)
    inject = all_inject.darrays[i].data
    cell = all_cells.darrays[i].data
    print('cell shape:', cell.shape)


    if i not in [61, 71, 72, 94, 98, 136]:

        centroid = get_weighted_centroid(inject, flatmap)

        centroid_idx = np.argmin(np.sum((flatmap - centroid) ** 2, axis=1))


        max_injection_idx = np.argmax(inject)
        max_site[i] = flatmap[max_injection_idx]


        print('Paxinos centroid label:', paxinos_area_name[paxinos_label[centroid_idx]])
        print('V4 centroid label:', v4_label[centroid_idx])
        print('Paxinos max label:', paxinos_area_name[paxinos_label[max_injection_idx]])
        print('V4 max label:', v4_label[max_injection_idx])

        if i in [91, 97]:

            injection_site[i] = flatmap[max_injection_idx]
            inject_area = v4_label[max_injection_idx]
        else:

            injection_site[i] = flatmap[centroid_idx]
            inject_area = v4_label[centroid_idx]
        inject_areas.append(inject_area)


        int_cell = cell.astype(int)
        distance_to_repeat = np.sqrt(((flatmap[:, 0] - injection_site[i][0])/xrange*xscale)**2+((flatmap[:, 1] - injection_site[i][1])/yrange*yscale)**2)
        print(distance_to_repeat.shape)
        local_distance = cf.repeat_data(distance_to_repeat, int_cell)
        distance.append(local_distance)


        scaled_flatmap = np.array([flatmap[:, 0]/xrange*xscale, flatmap[:, 1]/yrange*yscale]).T
        scaled_injection_site = np.array([injection_site[i][0]/xrange*xscale, injection_site[i][1]/yrange*yscale])
        angle_to_repeat = cf.get_angle(scaled_flatmap, scaled_injection_site)
        local_angle = cf.repeat_data(angle_to_repeat, int_cell)
        angle.append(local_angle)


        max_area = np.argmax(cell)
        print('Max voxel in Paxinos:', paxinos_area_name[paxinos_label[max_area]])
        print('Max voxel in V4:', v4_label[max_area])
        print('Max voxel value:', cell[max_area])

distance = np.concatenate(distance)
angle = np.concatenate(angle)

fig, axes = cf.get_fig_ax(ncols=2)

ax = axes[0]
cf.plt_hist(ax, distance, color=cf.PURPLE)
cf.set_ax(ax, xlabel='Distance 2D (mm)', ylog=True)

ax = axes[1]
rng = np.random.default_rng(0)
uniform_angle_data = rng.uniform(0., 2 * np.pi, len(angle))
cf.plt_compare_cdf_ks(ax, angle, uniform_angle_data, label1='data', label2='uniform', color1=cf.PURPLE, color2=cf.GREEN)
cf.set_ax(ax, xlabel='Angle')
