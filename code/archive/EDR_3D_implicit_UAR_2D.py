animal = 'marmoset'
dataset_key = 'majka'

animal_info = uf.AnimalInfo(animal, dataset_key)

area_center_dict_3D = animal_info.area_center_dict_3D

connection_df = animal_info.SC_data.copy()
connection_df_prob = animal_info.SC_data.copy()
seed = 0
rng = cf.get_local_rng(seed)
for source_area in area_center_dict_3D.keys():
    source_center = np.array(area_center_dict_3D[source_area])
    for target_area in area_center_dict_3D.keys():
        if source_area == target_area:
            continue
        target_center = np.array(area_center_dict_3D[target_area])
        distance = np.linalg.norm(source_center - target_center)
        connection_df.loc[source_area, target_area] = np.exp(-distance * 0.188)
        connection_df_prob.loc[source_area, target_area] = float(rng.uniform(0, 1) < (np.exp(-distance * 0.188) * 0.188))

angle_matrix = animal_info.angle_2D
fig, axes = cf.gfa(ncols=2)
ax = axes[0]
uf.visualize_data_uniform_distribution_repeated_by_SC(ax, angle_matrix, connection_df, minimal_tol=1e-5, distribution_type='CDF', distribution_kwargs={'show_list': ['KS']})
cf.set_ax(ax, xlabel='Angle', ylabel='CDF', legend_loc='lower right', title='weighted')

ax = axes[1]
uf.visualize_data_uniform_distribution_repeated_by_SC(ax, angle_matrix, connection_df_prob, minimal_tol=1e-5, distribution_type='CDF', distribution_kwargs={'show_list': ['KS']})
cf.set_ax(ax, xlabel='Angle', ylabel='CDF', legend_loc='lower right', title='binary')