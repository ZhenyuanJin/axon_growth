import copy
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.feature_selection import mutual_info_regression

from utils_function import cf
from utils_function import utils_function as uf


def get_multi_panel_figure(title_list, layout='column', subfig_nrows=None, subfig_ncols=None, get_suitable_fig_size_kwargs=None, get_fig_subfig_kwargs=None, get_ax_kwargs=None):
    num = len(title_list)
    if subfig_nrows is None and subfig_ncols is None:
        if layout == 'row':
            subfig_nrows, subfig_ncols = num, 1
        else:
            subfig_nrows, subfig_ncols = 1, num
    elif subfig_nrows is None:
        subfig_nrows = int(np.ceil(num / subfig_ncols))
    elif subfig_ncols is None:
        subfig_ncols = int(np.ceil(num / subfig_nrows))
    get_suitable_fig_size_kwargs = cf.update_dict({}, get_suitable_fig_size_kwargs)
    get_fig_subfig_kwargs = cf.update_dict({}, get_fig_subfig_kwargs)
    get_ax_kwargs = cf.update_dict({}, get_ax_kwargs)
    width, height = cf.get_suitable_fig_size(**get_suitable_fig_size_kwargs)
    fig = plt.figure(figsize=(width * subfig_ncols, height * subfig_nrows), dpi=cf.FIG_DPI)
    subfigures = np.asarray(fig.subfigures(subfig_nrows, subfig_ncols, squeeze=False, **get_fig_subfig_kwargs), dtype=object).flatten()
    subfig_by_name = {}
    axes_by_name = {}
    for index, title in enumerate(title_list):
        subfig_by_name[title] = subfigures[index]
        axes_by_name[title] = cf.get_ax(fig=subfigures[index], **get_ax_kwargs)
    return fig, subfig_by_name, axes_by_name


def add_multi_panel_titles(subfig_by_name, title_list, figure_title_list, y=0.98):
    for title, figure_title in zip(title_list, figure_title_list):
        subfig_by_name[title].suptitle(cf.format_text(figure_title), y=y, fontsize=cf.SUP_TITLE_SIZE)


def add_multi_panel_tags(axes_by_name, title_list, layout='column', row_first=True, **kwargs):
    axes = [list(np.asarray(axes_by_name[title], dtype=object).reshape(-1)) for title in title_list]
    if row_first and layout == 'column':
        ordered_axes = [ax for row in zip(*axes) for ax in row]
    else:
        ordered_axes = [ax for group in axes for ax in group]
    cf.add_axes_list_tag_by_order(ordered_axes, **kwargs)


def plot_connection_on_flatmap(ax, animal, dataset_key, connection_matrix, which_area, connection_matrix_vmax, focus_direction, source_names, target_names, data_label, show_value_on_flatmap_kwargs=None):
    show_value_on_flatmap_kwargs = cf.update_dict({}, show_value_on_flatmap_kwargs)
    value_dict = {}
    target = None
    source = None
    if focus_direction == 'col':
        target = which_area
        cmap = cf.get_cmap([cf.TRANSPARENT, uf.SOURCE_COLOR])
        for source_name in source_names:
            value_dict[source_name] = connection_matrix.loc[source_name, which_area]
    elif focus_direction == 'row':
        source = which_area
        cmap = cf.get_cmap([cf.TRANSPARENT, uf.TARGET_COLOR])
        for target_name in target_names:
            value_dict[target_name] = connection_matrix.loc[which_area, target_name]
    else:
        raise ValueError(f'Unknown focus direction: {focus_direction}')
    uf.plot_show_value_on_flatmap(ax, animal=animal, dataset_key=dataset_key, value_dict=value_dict, cmap=cmap, cbar_label='SC', source=source, target=target, vmax=connection_matrix_vmax, **show_value_on_flatmap_kwargs)
    cf.set_ax(ax, title=data_label + '\n' + which_area)


def plot_connection_by_arrows_on_flatmap(ax, animal, dataset_key, connection_matrix, selected_target_area_list, data_label, show_area_name=False, text_fontsize=None, show_connection_on_flatmap_kwargs=None):
    if text_fontsize is None:
        text_fontsize = cf.FONT_SIZE / 3
    show_connection_on_flatmap_kwargs = cf.update_dict({}, show_connection_on_flatmap_kwargs)
    uf.plot_show_connection_on_flatmap(ax, animal=animal, dataset_key=dataset_key, SC=connection_matrix, selected_target_area_list=selected_target_area_list, show_area_name=show_area_name, text_fontsize=text_fontsize, **show_connection_on_flatmap_kwargs, edgecolor=(0, 0, 0, 0.3))
    cf.set_ax(ax, title=data_label)


def plot_data_model_connection_matrices(ax_data, ax_model, data_matrix, model_matrix, animal, dataset_key, topo=False, show_correlation=True):
    uf.plot_visualize_connection_matrix_heatmap(ax_data, ax_model, matrix_data=data_matrix, matrix_model=model_matrix, animal=animal, dataset_key=dataset_key, log_matrix=False, topo=topo, show_correlation=show_correlation)


def plot_area_level_performance(ax, data_matrix, model_matrix, animal, dataset_key, sort=True, diagonal=False):
    performance_dict = uf.area_level_correlation(SC_data=data_matrix, SC_model=model_matrix, animal=animal, dataset_key=dataset_key, diagonal=diagonal)
    if sort:
        performance_dict = cf.sort_dict(performance_dict, reverse=True)
    cf.plt_bar_dict(ax, data=performance_dict, color=cf.BLUE)
    cf.add_hline(ax, y=np.nanmean(list(performance_dict.values())), color=cf.GREEN, linestyle='--', label='mean')
    cf.set_ax(ax, xlabel='Area', ylabel='correlation', xtick_rotation=90)


def plot_distance_distribution(ax, distance_matrix, connection_matrix, minimal_tol, data_label, dim='2D', distribution_kwargs=None, linregress=False, linregress_kwargs=None):
    color = uf.MODEL_COLOR if data_label == 'model' else uf.DATA_COLOR
    distribution_kwargs = cf.update_dict({'color': color}, distribution_kwargs)
    histogram = uf.plot_visualize_distribution_repeated_by_SC(ax, distance_matrix, connection_matrix, minimal_tol=minimal_tol, distribution_type='hist', distribution_kwargs=distribution_kwargs)
    if linregress:
        linregress_kwargs = cf.update_dict({}, linregress_kwargs)
        regress_dict = cf.plt_linregress(ax, x=histogram.mid_point, y=histogram.hist, ylog=True, line_color=cf.BLACK, show_scatter=False, show_list=[], **linregress_kwargs)
        regress_dict[r'$\Lambda$'] = -1 / (regress_dict['slope'] * np.log(10))
        unit_dict = cf.create_dict(keys=regress_dict.keys(), default_value=None)
        unit_dict[r'$\Lambda$'] = 'mm'
        cf.add_text_by_dict(ax, regress_dict, show_list=['p', r'$\Lambda$'], unit_dict=unit_dict, text_x=0.45, text_y=0.9, fontsize=cf.LABEL_SIZE * 0.75)
    cf.set_ax(ax, xlabel=f'Distance {dim} (mm)', ylog=True, title=data_label)


def plot_angle_distribution(ax, connection_matrix, minimal_tol, data_label, dim='2D', angle_matrix=None, azimuth_matrix=None, elevation_matrix=None, distribution_kwargs=None):
    color = uf.MODEL_COLOR if data_label == 'model' else uf.DATA_COLOR
    if dim == '2D':
        distribution_kwargs = cf.update_dict({'color': color}, distribution_kwargs)
        uf.plot_visualize_distribution_repeated_by_SC(ax, angle_matrix, connection_matrix, minimal_tol=minimal_tol, distribution_type='CDF', distribution_kwargs=distribution_kwargs)
        cf.set_ax(ax, xlabel='Angle', ylabel='CDF', legend_loc='lower right')
        return None
    connection_int = uf.scale_SC_to_int(connection_matrix, minimal_tol=minimal_tol)
    azimuth_values = cf.repeat_data(azimuth_matrix, connection_int)
    elevation_values = cf.repeat_data(elevation_matrix, connection_int)
    angle_heatmap, *_ = cf.get_hist_2d(azimuth_values, elevation_values, cf.BIN_NUM, cf.BIN_NUM)
    cf.sns_marginal_heatmap(ax, angle_heatmap, outside=False, heatmap_kwargs={'cmap': cf.get_cmap([cf.WHITE, color]), 'show_xtick': False, 'show_ytick': False, 'vmin': 0.0, 'vmax': 0.02, 'cbar_kwargs': {'add_geq': True}}, x_color=color, y_color=color)
    cf.set_ax(ax, xlabel='Azimuth', ylabel='Elevation', title=data_label)
    return angle_heatmap


def plot_data_model_angle_distribution(axes, data_matrix, model_matrix, minimal_tol, azimuth_matrix, elevation_matrix):
    data_heatmap = plot_angle_distribution(axes[0], data_matrix, minimal_tol, 'data', dim='3D', azimuth_matrix=azimuth_matrix, elevation_matrix=elevation_matrix)
    model_heatmap = plot_angle_distribution(axes[1], model_matrix, minimal_tol, 'model', dim='3D', azimuth_matrix=azimuth_matrix, elevation_matrix=elevation_matrix)
    jsd = cf.get_jsd(data_heatmap.flatten(), model_heatmap.flatten())
    axes[0].set_title('')
    axes[1].set_title('')
    cf.add_text(axes[1], x=0.5, y=1.7, text=f'JSD={round(jsd, 3)}', va='center', ha='center', fontsize=cf.LABEL_SIZE)


def plot_data_uniform_angle_distribution(ax, angle_matrix, data_matrix, minimal_tol):
    uf.plot_visualize_data_uniform_distribution_repeated_by_SC(ax, angle_matrix, data_matrix, seed=0, minimal_tol=minimal_tol, distribution_type='CDF', distribution_kwargs={'show_list': ['KS']})
    cf.set_ax(ax, xlabel='Angle', ylabel='CDF', legend_loc='lower right')


def plot_angle_uniformity_by_distance_threshold(ax, distance_matrix, angle_matrix, data_matrix, minimal_tol, distance_threshold_list, lambda_value):
    data_int = uf.scale_SC_to_int(data_matrix.copy(), minimal_tol=minimal_tol)
    results = []
    valid_distances = []
    for distance_threshold in distance_threshold_list:
        try:
            masked_data = data_int.copy()
            masked_data[distance_matrix < distance_threshold] = 0
            repeated_angles = cf.repeat_data(angle_matrix, masked_data)
            uniform_cdf = lambda x: scipy.stats.uniform.cdf(x, loc=-np.pi, scale=2 * np.pi)
            results.append(cf.get_ks_result(repeated_angles, uniform_cdf))
            valid_distances.append(distance_threshold)
        except Exception:
            pass
    cf.plt_scatter(ax, x=valid_distances, y=[result['ks'] for result in results], marker='o', color=cf.BLUE)
    cf.add_vline(ax, x=lambda_value * 5, color=cf.RED, linestyle='--', label=r'$5\Lambda$')
    cf.set_ax(ax, xlabel='threshold (mm)', ylabel='KS', legend_loc='lower right')


def plot_intrinsic_proportion(ax, target_names, clustered_target_names, data_proportion, model_matrix, animal, dataset_key):
    model_proportion = uf.get_intrinsic_proportion_from_SCt(model_matrix, animal, dataset_key)
    data_values = [data_proportion[area] for area in clustered_target_names]
    model_values = [model_proportion[area] for area in clustered_target_names]
    x = range(len(target_names))
    cf.plt_two_side_bar(ax, x, data_values, model_values, color1=uf.DATA_COLOR, color2=uf.MODEL_COLOR, label1='data', label2='model')
    cf.add_text(ax, text='Corr=' + str(round(cf.get_corr(data_values, model_values), 3)), fontsize=cf.LABEL_SIZE)
    cf.set_ax(ax, xlabel='Area', ylabel='Intrinsic proportion', xtick=x, xtick_label=target_names, xtick_rotation=90, ylim=[-1, 1])


def plot_distance_connection_scatter(ax, distance_matrix, connection_matrix, animal, dataset_key, data_label, dim='2D'):
    logged_connection = uf.log_SC(connection_matrix, animal, dataset_key, apply_c=False, apply_w=False)
    distances, connections = cf.sync_special_value(distance_matrix.values.flatten(), logged_connection.values.flatten())
    distances = cf.process_nan(distances, nan_policy='drop')
    connections = cf.process_nan(connections, nan_policy='drop')
    cf.plt_density_scatter(ax, distances, connections, cmap=plt.cm.coolwarm, cbar_kwargs={'ticks': []})
    cf.set_ax(ax, xlabel=f'Distance {dim} (mm)', ylabel='log(SC)', title=data_label)


def plot_connection_distribution(ax, connection_matrix, data_label, log_scale=True, distribution_kwargs=None):
    color = uf.MODEL_COLOR if data_label == 'model' else uf.DATA_COLOR
    distribution_kwargs = cf.update_dict({'color': color}, distribution_kwargs)
    cf.sns_hist(ax, data=connection_matrix.values.flatten(), bins=cf.BIN_NUM, log_scale=log_scale, **distribution_kwargs)
    cf.set_ax(ax, xlabel='SC', title=data_label)


def plot_source_points_by_target_position(ax, source_points, target_points, position, radius, animal, dataset_key):
    target_distances = np.linalg.norm(target_points - position, axis=1)
    selected_source_points = source_points[np.where(target_distances < radius)[0]]
    cf.plt_scatter(ax, selected_source_points[:, 0], selected_source_points[:, 1], color=cf.BLACK, label='source point', s=5)
    uf.plot_show_area_boundary(ax, animal=animal, dataset_key=dataset_key, show_area_name=False, edgecolor=cf.BLACK)
    cf.add_circle_patch(ax, center=position, radius=radius, edgecolor=cf.WHITE, facecolor=cf.WHITE)


def plot_removed_long_range_correlation(ax, distance_matrix, data_matrix, model_matrix, animal, dataset_key, lambda_value, removed_value, point_num=100, removal_mode='data'):
    logged_model = uf.log_SC(model_matrix, animal, dataset_key, apply_c=True, apply_w=True)
    logged_data = uf.log_SC(data_matrix, animal, dataset_key, apply_c=True, apply_w=True)
    correlations = uf.get_rm_longrange_correlation_change(logged_data, logged_model, distance_matrix, animal, dataset_key, removed_value, point_num=point_num, rm_mode=removal_mode)
    cf.plt_line(ax, x=np.linspace(0, np.nanmax(distance_matrix.values), point_num), y=correlations, color=cf.ORANGE)
    cf.add_vline(ax, x=lambda_value * 5, color=cf.RED, linestyle='--', label=r'$5\Lambda$')
    cf.set_ax(ax, xlabel='Threshold (mm)', ylabel='Correlation', legend_loc='lower right')


def plot_category_long_range_connection(ax, long_range_connection_sum):
    categories = uf.get_area_category_in_order()
    cf.plt_bar(ax, range(len(categories)), [long_range_connection_sum[category] for category in categories])
    cf.set_ax(ax, ylabel='Long range sum', xtick=range(len(categories)), xtick_label=categories, xtick_rotation=90)


def plot_injection_deviation_performance(ax, deviation_by_area, performance_by_area):
    x = [deviation_by_area[area] for area in performance_by_area]
    y = [performance_by_area[area] for area in performance_by_area]
    cf.plt_linregress(ax, x, y, show_list=['p', 'r'], text_kwargs={'text_x': 0.02, 'text_y': 0.2}, text_size=cf.LABEL_SIZE * 0.75)
    cf.add_text(ax, x=deviation_by_area['A19M'], y=performance_by_area['A19M'], text='A19M', va='bottom', ha='center', fontsize=cf.LABEL_SIZE * 0.75, transform='data')
    cf.set_ax(ax, xlabel='Injection deviation level', ylabel='correlation')


def plot_direction_preference_performance(ax, preference_by_area, performance_by_area):
    x = [preference_by_area[area] for area in performance_by_area]
    y = [performance_by_area[area] for area in performance_by_area]
    cf.plt_linregress(ax, x, y, show_list=['p', 'r'], text_kwargs={'text_x': 0.02, 'text_y': 0.2}, text_size=cf.LABEL_SIZE * 0.75)
    cf.set_ax(ax, xlabel='Direction preference', ylabel='correlation')


def plot_2d_3d_distance_scatter(ax, distance_2d, distance_3d):
    reordered_distance_3d = cf.reorder_df(distance_3d, new_row=distance_2d.index, new_col=distance_2d.columns)
    cf.plt_linregress(ax, reordered_distance_3d.values.flatten(), distance_2d.values.flatten(), show_list=['r'], scatter_mode='density', scatter_kwargs={'cbar_kwargs': {'ticks': []}}, text_size=cf.LABEL_SIZE * 0.75, text_kwargs={'text_x': 0.05, 'text_y': 1.0})
    cf.set_ax(ax, xlabel='Distance 3D (mm)', ylabel='Distance 2D (mm)')


def get_angle_distance_values(weight_mask, angle_matrix, distance_matrix, distance_threshold=None):
    angles = []
    distances = []
    for row in weight_mask.index:
        for column in weight_mask.columns:
            keep = weight_mask.loc[row, column]
            if distance_threshold is not None:
                keep = keep and distance_matrix.loc[row, column] <= distance_threshold
            if keep:
                angles.append(angle_matrix.loc[row, column])
                distances.append(distance_matrix.loc[row, column])
    return np.asarray(angles), np.asarray(distances)


def plot_angle_distance_scatter(ax, weight_mask, angle_matrix, distance_matrix, n_neighbors):
    angles, distances = get_angle_distance_values(weight_mask, angle_matrix, distance_matrix)
    cf.plt_scatter(ax, distances, angles)
    mutual_info = mutual_info_regression(distances.reshape(-1, 1), angles, n_neighbors=n_neighbors, random_state=0)[0]
    cf.add_text_by_dict(ax, {'mutual_info': mutual_info}, text_x=0.5, text_y=1.0, text_kwargs={'transform': ax.transAxes, 'ha': 'center', 'va': 'bottom'}, fontsize=cf.LABEL_SIZE * 0.75)
    cf.set_ax(ax, xlabel='Distance (mm)', ylabel='Angle')


def plot_angle_distance_dependence(ax, weight_mask, angle_matrix, distance_matrix, distance_threshold_list, n_neighbors, lambda_value, line_xlim=None):
    mutual_information = []
    for distance_threshold in distance_threshold_list:
        angles, distances = get_angle_distance_values(weight_mask, angle_matrix, distance_matrix, distance_threshold)
        try:
            value = mutual_info_regression(distances.reshape(-1, 1), angles, n_neighbors=n_neighbors, random_state=0)[0]
        except Exception:
            value = np.nan
        mutual_information.append(value)
    cf.plt_line(ax, x=distance_threshold_list, y=mutual_information, color=cf.BLACK, label='Mutual info', xlim=line_xlim)
    cf.add_vline(ax, x=lambda_value * 5, color=cf.RED, linestyle='--', label=r'$5\Lambda$')
    cf.set_ax(ax, xlabel='Threshold (mm)', ylabel='Mutual info', legend_loc='upper left')


def plot_distance_relative_error(ax, distance_matrix, relative_error, data_matrix, source_names, target_names, dim='2D'):
    distances = []
    errors = []
    for source in source_names:
        for target in target_names:
            if source != target and data_matrix.loc[source, target] > 0:
                distances.append(distance_matrix.loc[source, target])
                errors.append(relative_error.loc[source, target])
    cf.plt_smooth_scatter_line(ax, distances, errors)
    cf.set_ax(ax, xlabel=f'Distance {dim} (mm)', ylabel='Relative error', legend_loc='lower left', bbox_to_anchor=(0, 0.95))


def plot_scaling_law(ax, animal_list, distance_matrix_list, lambda_list):
    max_distances = [np.nanmax(matrix.values) for matrix in distance_matrix_list]
    cf.plt_linregress(ax, x=max_distances, y=lambda_list, xlog=True, ylog=True, show_list=['slope'], scatter_kwargs={'marker': '*', 's': 1000.0})
    for animal, x, y in zip(animal_list, max_distances, lambda_list):
        cf.add_text(ax, x=x * 1.2, y=y, text=animal, va='top', ha='left', transform='data')
    cf.set_ax(ax, xlabel='Max distance (mm)', ylabel=r'Optimal $\Lambda$ (mm)', xlog=True, ylog=True)


def plot_long_range_correlation_heatmap(ax, animal_list, long_range_sum_list):
    categories = uf.get_area_category_in_order()
    arrays = [np.array([values[category] for category in categories]) for values in long_range_sum_list]
    correlation = pd.DataFrame(index=animal_list, columns=animal_list, dtype=float)
    for row, row_values in zip(animal_list, arrays):
        for column, column_values in zip(animal_list, arrays):
            correlation.loc[row, column] = np.corrcoef(row_values, column_values)[0, 1]
    cf.sns_heatmap(ax, correlation, cmap=cf.get_cmap([cf.WHITE, cf.RED]), vmin=0, vmax=1, cbar_label='Correlation', show_xtick=True, show_ytick=True, heatmap_kwargs={'annot': True, 'fmt': '.2f'})


class PaperFig:
    def __init__(self, model_timedir_list, marmoset_V4_timedir, output_dir=None):
        self.animal_list = ['mouse', 'marmoset', 'macaque', 'human']
        self.dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere']
        self.dataset_name_list = [cf.cat(animal, dataset_key) for animal, dataset_key in zip(self.animal_list, self.dataset_key_list)]
        self.model_timedir_list = model_timedir_list
        self.animal_info_by_name = {}
        self.model_by_name = {}
        self.data_matrix_by_name = {}
        self.model_matrix_by_name = {}
        self.masked_model_matrix_by_name = {}
        self.minimal_tol_by_name = {}
        for animal, dataset_key, model_timedir in zip(self.animal_list, self.dataset_key_list, model_timedir_list):
            self._load_dataset(animal, dataset_key, model_timedir)
        self._load_dataset('marmoset', 'V4', marmoset_V4_timedir)
        if output_dir is None:
            output_dir = Path(__file__).resolve().parents[2] / 'results' / 'figures'
        self.figdir = str(output_dir)
        self.formats = [cf.SAVEFIG_RASTER_FORMAT, cf.SAVEFIG_FORMAT]
        self.save_fig_kwargs = {'formats': self.formats, 'bbox_inches': 'tight'}

    def _load_dataset(self, animal, dataset_key, model_timedir):
        name = cf.cat(animal, dataset_key)
        animal_info = uf.AnimalInfo(animal, dataset_key)
        model = uf.ConnectionGenerationModel()
        model.set_timedir(model_timedir, load_simulation=False)
        data_matrix = animal_info.SC_data
        positive_values = data_matrix.values[data_matrix.values > 0]
        self.animal_info_by_name[name] = animal_info
        self.model_by_name[name] = model
        self.data_matrix_by_name[name] = data_matrix
        self.model_matrix_by_name[name] = model.get_suitable_SC_model()
        self.masked_model_matrix_by_name[name] = model.get_suitable_SC_model_masked_as_data()
        self.minimal_tol_by_name[name] = np.min(positive_values)

    def plot_all(self):
        plot_methods = [
            self.plot_connection_generation_method_overview,
            self.plot_cross_species_connectivity_fit,
            self.plot_direction_distribution_comparison,
            self.plot_marmoset_connection_statistics,
            self.plot_long_range_connection_analysis,
            self.plot_cross_species_scaling,
            self.plot_distance_2d_3d_and_edr,
            self.plot_angle_distance_dependence,
            self.plot_marmoset_atlas_comparison,
            self.plot_connection_statistics_other_species,
            self.plot_marmoset_area_performance_and_retrograde,
            self.plot_area_performance_all_species,
            self.plot_distance_relative_error,
            self.plot_angle_uniformity_by_distance_threshold,
            self.plot_marmoset_direction_and_injection_bias,
            self.plot_direction_preference_performance,
        ]
        for plot_method in plot_methods:
            plot_method()
            gc.collect()

    def plot_connection_generation_method_overview(self):
        animal = 'marmoset'
        dataset_key = 'majka'
        name = 'marmoset_majka'
        animal_info = self.animal_info_by_name[name]
        point_density = 15
        lambda_value = 2
        margin = cf.COMPACT_MARGIN.copy()
        margin['top'] = 0.87
        margin['bottom'] = 0.13
        fig, axes = cf.get_fig_ax(ncols=3, nrows=2, margin=margin, ax_width=cf.AX_WIDTH * 1.15, ax_height=cf.AX_HEIGHT * 1.15)
        axes_by_tag = {'a': axes[0, 0], 'b': axes[0, 1], 'c': axes[0, 2], 'd': axes[1, 0], 'e': axes[1, 1], 'f': axes[1, 2]}
        cf.add_axes_dict_tag(axes_dict=axes_by_tag, ref_ax=axes[0, 0])

        cf.set_ax(axes_by_tag['a'], title='Cortical surface')
        uf.plot_show_whole_cortex_boundary(axes_by_tag['a'], animal, dataset_key)

        cf.set_ax(axes_by_tag['b'], title='Neuron generation')
        uf.plot_show_whole_cortex_boundary(axes_by_tag['b'], animal, dataset_key)
        scatter_size = cf.get_suitable_s(axes_by_tag['b'], num=int(point_density * animal_info.whole_cortex_polygon.area)) / 10
        for polygon in animal_info.polygon_dict.values():
            points = uf.get_random_points_in_polygon(polygon, density=point_density, seed=0)
            cf.plt_scatter(axes_by_tag['b'], points[:, 0], points[:, 1], color=cf.RYO, s=scatter_size, zorder=-1, alpha=0.5)

        ax = axes_by_tag['c']
        cf.set_ax(ax, title='Connection generation methodology')
        cf.add_text(ax, text=r'$\theta \sim \mathrm{Uniform}(-\pi, \pi)$', x=0.5, y=0.0, va='center', ha='center', fontsize=cf.LABEL_SIZE)
        cf.add_text(ax, text=r'$d \sim \mathrm{Exponential}(\Lambda)$', x=0.5, y=-0.08, va='center', ha='center', fontsize=cf.LABEL_SIZE, text_process={'capitalize': False})
        uf.plot_show_whole_cortex_boundary(ax, animal, dataset_key)
        scatter_size = cf.get_suitable_s(ax, num=int(point_density * animal_info.whole_cortex_polygon.area)) / 10
        for polygon in animal_info.polygon_dict.values():
            points = uf.get_random_points_in_polygon(polygon, density=point_density, seed=0)
            cf.plt_scatter(ax, points[:, 0], points[:, 1], color=cf.RYO, s=scatter_size, zorder=-1, alpha=0.5)
        source_point = np.array([-8.2, 1.6])
        angle = 45 / 180 * np.pi
        distance = lambda_value * 5
        target_point = source_point + np.array([np.cos(angle) * distance, np.sin(angle) * distance])
        cf.plt_scatter(ax, [source_point[0]], [source_point[1]], color=cf.CRIMSON, s=50, zorder=1)
        cf.add_arrow(ax, x_start=source_point[0], y_start=source_point[1], x_end=target_point[0], y_end=target_point[1], fc=cf.CRIMSON, ec=cf.CRIMSON)
        cf.add_arc_patch(ax, source_point, distance * 0.3, 0, angle, facecolor=cf.CRIMSON, edgecolor=cf.CRIMSON)
        cf.plt_line(ax, [source_point[0], source_point[0] + distance], [source_point[1], source_point[1]], color=cf.CRIMSON, linestyle='--')
        cf.add_text(ax, text='$\\theta$', x=source_point[0] + np.cos(angle / 2) * distance * 0.4, y=source_point[1] + np.sin(angle / 2) * distance * 0.4, va='center', ha='center', transform='data', fontsize=cf.LABEL_SIZE)
        cf.add_text(ax, text='d', x=(source_point[0] + target_point[0]) / 2, y=(source_point[1] + target_point[1]) / 2, va='bottom', ha='right', transform='data', text_process={'capitalize': False}, fontsize=cf.LABEL_SIZE)

        source_points = [[-8.2, 1.6], [4.0, 3.5], [-5.0, 3.0], [6.7, 2.5], [-13, 5.0], [9.8, 6.0], [10.5, 7.0], [7.0, -4.0], [0.0, -5.0]]
        angles = np.array([45, 230, 270, 315, 10, 170, 120, -10, -30]) / 180 * np.pi
        distances = np.array([5, 3, 7, 4, 1.5, 2.5, 1.8, 2.9, 2.2]) * lambda_value
        for tag, title, show_areas in [('d', 'Connection generation', False), ('e', 'Atlas registration', True)]:
            ax = axes_by_tag[tag]
            cf.set_ax(ax, title=title)
            if show_areas:
                uf.plot_show_area_boundary(ax, animal, dataset_key)
            else:
                uf.plot_show_whole_cortex_boundary(ax, animal, dataset_key)
            for local_source, local_angle, local_distance in zip(source_points, angles, distances):
                local_target = local_source + np.array([np.cos(local_angle) * local_distance, np.sin(local_angle) * local_distance])
                cf.add_arrow(ax, x_start=local_source[0], y_start=local_source[1], x_end=local_target[0], y_end=local_target[1], fc=cf.CRIMSON, ec=cf.CRIMSON)
            scatter_size = cf.get_suitable_s(ax, num=int(point_density * animal_info.whole_cortex_polygon.area)) / 10
            for polygon in animal_info.polygon_dict.values():
                points = uf.get_random_points_in_polygon(polygon, density=point_density, seed=0)
                cf.plt_scatter(ax, points[:, 0], points[:, 1], color=cf.RYO, s=scatter_size, zorder=-1, alpha=0.5)

        plot_connection_by_arrows_on_flatmap(axes_by_tag['f'], animal, dataset_key, self.model_matrix_by_name[name], animal_info.source_name, 'model')
        axes_by_tag['f'].set_title('Area level connection')
        cf.save_fig(fig, self.figdir + '/connection_generation_method_overview', **self.save_fig_kwargs)

    def plot_cross_species_connectivity_fit(self):
        fig, axes = cf.get_fig_ax(nrows=3, ncols=len(self.animal_list))
        performance_dict = {}
        processed_performance_dict = {}
        for index, (animal, dataset_key, name) in enumerate(zip(self.animal_list, self.dataset_key_list, self.dataset_name_list)):
            data_matrix = self.data_matrix_by_name[name]
            model_matrix = self.masked_model_matrix_by_name[name]
            plot_data_model_connection_matrices(axes[0, index], axes[1, index], data_matrix, self.model_matrix_by_name[name], animal, dataset_key)
            cf.add_axes_title(axes[0, index], title=animal, title_size=cf.SUP_TITLE_SIZE)
            performance_dict[name] = uf.get_performance_dict(SC_data=data_matrix, SC_model=model_matrix, animal=animal, dataset_key=dataset_key, diagonal=False)
            processed_performance_dict[name] = performance_dict[name].copy()
            for key in ['correlation', 'accuracy_all', 'accuracy_non', 'accuracy_both']:
                processed_performance_dict[name].pop(key, None)
        for name, values in processed_performance_dict.items():
            processed_performance_dict[name] = {key.replace('accuracy_', '') if key.startswith('accuracy_') else key: value for key, value in values.items()}
        merged_ax = cf.merge_ax(axes[2, :])
        cf.plt_group_bar_dict(merged_ax, processed_performance_dict, width=0.05)
        cf.set_ax(merged_ax, xtick=range(len(self.animal_list)), xtick_label=self.animal_list, ylim=[0, 1], legend_kwargs={'ncols': 6}, legend_loc='lower right', bbox_to_anchor=(1, 1), xtick_size=cf.TITLE_SIZE * 1.2, adjust_tick_size=False, legend_size=cf.TITLE_SIZE, ylabel='Performance', label_size=cf.TITLE_SIZE * 1.2, ytick_size=cf.LABEL_SIZE)
        cf.add_axes_list_tag_by_order([axes[0, index] for index in range(len(self.animal_list))] + [axes[2, 0]])
        cf.save_fig(fig, self.figdir + '/cross_species_connectivity_fit', **self.save_fig_kwargs)
        cf.save_dict(performance_dict, self.figdir + '/fig_2_performance_dict')
        relative_error_dict = copy.deepcopy(performance_dict)
        for name, values in performance_dict.items():
            for key, value in values.items():
                if key.startswith('accuracy_'):
                    relative_error_dict[name][key.replace('accuracy_', 'RMAE_')] = 1 - value
                    relative_error_dict[name].pop(key)
        cf.save_dict(relative_error_dict, self.figdir + '/fig_2_RMAE_dict')

    def plot_direction_distribution_comparison(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 3}, get_suitable_fig_size_kwargs={'nrows': 3, 'which': 'adjust_params_custom'})
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            axes = axes_by_name[name]
            plot_data_uniform_angle_distribution(axes[0], animal_info.angle_2D, self.data_matrix_by_name[name], self.minimal_tol_by_name[name])
            plot_data_model_angle_distribution(axes[1:], self.data_matrix_by_name[name], self.masked_model_matrix_by_name[name], self.minimal_tol_by_name[name], animal_info.angle_3D['azim'], animal_info.angle_3D['elev'])
        first_axes = axes_by_name[self.dataset_name_list[0]]
        cf.plt_bar(first_axes[1], [0], [0], label='Data', color=uf.DATA_COLOR)
        cf.set_ax(first_axes[1], legend_loc='lower left', bbox_to_anchor=(0, 1.4))
        cf.plt_bar(first_axes[2], [0], [0], label='Model', color=uf.MODEL_COLOR)
        cf.set_ax(first_axes[2], legend_loc='lower left', bbox_to_anchor=(0, 1.4))
        add_multi_panel_tags(axes_by_name, self.dataset_name_list, row_first=False, ref_ax=first_axes[0])
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list)
        cf.save_fig(fig, self.figdir + '/direction_distribution_comparison', **self.save_fig_kwargs)

    def plot_marmoset_connection_statistics(self):
        name = 'marmoset_majka'
        animal = 'marmoset'
        dataset_key = 'majka'
        animal_info = self.animal_info_by_name[name]
        model = self.model_by_name[name]
        fig, axes = cf.get_fig_ax(nrows=2, ncols=4)
        axes_by_tag = {'a': cf.merge_ax(axes[0, 0:2]), 'b': cf.merge_ax(axes[0, 2]), 'c': cf.merge_ax(axes[0, 3]), 'd': cf.merge_ax(axes[1, 0]), 'e': cf.merge_ax(axes[1, 1]), 'f': cf.merge_ax(axes[1, 2]), 'g': cf.merge_ax(axes[1, 3])}
        cf.add_axes_dict_tag(axes_dict=axes_by_tag, ref_ax=axes[0, 0])
        plot_intrinsic_proportion(axes_by_tag['a'], animal_info.target_name, animal_info.clustered_target_name, animal_info.plot_intrinsic_proportion, model.SC_model_dict['FLNt'], animal, dataset_key)
        plot_connection_distribution(axes_by_tag['b'], self.data_matrix_by_name[name], 'data')
        plot_connection_distribution(axes_by_tag['c'], self.masked_model_matrix_by_name[name], 'model')
        plot_distance_connection_scatter(axes_by_tag['d'], animal_info.distance_2D, self.data_matrix_by_name[name], animal, dataset_key, 'data')
        plot_distance_connection_scatter(axes_by_tag['e'], animal_info.distance_2D, self.masked_model_matrix_by_name[name], animal, dataset_key, 'model')
        plot_distance_distribution(axes_by_tag['f'], animal_info.distance_2D, self.data_matrix_by_name[name], self.minimal_tol_by_name[name], 'data')
        plot_distance_distribution(axes_by_tag['g'], animal_info.distance_2D, self.masked_model_matrix_by_name[name], self.minimal_tol_by_name[name], 'model')
        cf.save_fig(fig, self.figdir + '/marmoset_connection_statistics', **self.save_fig_kwargs)

    def plot_marmoset_area_performance_and_retrograde(self):
        name = 'marmoset_majka'
        animal = 'marmoset'
        dataset_key = 'majka'
        animal_info = self.animal_info_by_name[name]
        model = self.model_by_name[name]
        margin = cf.MARGIN.copy()
        margin['left'] -= 0.05
        margin['right'] += 0.05
        margin['top'] += 0.05
        margin['bottom'] -= 0.05
        fig, axes = cf.get_fig_ax(nrows=2, ncols=4, margin=margin)
        fig.set_figwidth(fig.get_figwidth() * 0.928435)
        fig.set_figheight(fig.get_figheight() * 1.002763)
        axes_by_tag = {'a': cf.merge_ax(axes[0, 0:2]), 'b': cf.merge_ax(axes[1, 0:2]), 'c': cf.merge_ax(axes[0, 2]), 'd': cf.merge_ax(axes[0, 3]), 'e': cf.merge_ax(axes[1, 2]), 'f': cf.merge_ax(axes[1, 3])}
        for tag, ax in axes_by_tag.items():
            if tag in ['a', 'b']:
                cf.move_ax(ax, d_right=-0.1, keep_size=False)
            elif tag in ['c', 'e']:
                cf.move_ax(ax, d_left=0.1, keep_size=True)
            else:
                cf.move_ax(ax, d_left=0.13, keep_size=True)
        cf.add_axes_dict_tag(axes_dict=axes_by_tag, ref_ax=axes[0, 0])
        plot_area_level_performance(axes_by_tag['a'], self.data_matrix_by_name[name], self.model_matrix_by_name[name], animal, dataset_key)
        cf.set_ax(axes_by_tag['a'], ylabel='Correlation (FLNe)', ylim=[-0.2, 1.2], xtick_rotation=90)
        plot_area_level_performance(axes_by_tag['b'], animal_info.SCt_data, model.SC_model_dict['FLNt'], animal, dataset_key, diagonal=True)
        cf.set_ax(axes_by_tag['b'], ylabel='Correlation (FLNt)', ylim=[-0.2, 1.2], xtick_rotation=90)
        plot_connection_on_flatmap(axes_by_tag['c'], animal, dataset_key, self.data_matrix_by_name[name], 'A1-2', animal_info.connection_matrix_vmax, animal_info.SC_focus_direction, animal_info.source_name, animal_info.target_name, 'data', {'cbar': False})
        plot_connection_on_flatmap(axes_by_tag['d'], animal, dataset_key, self.masked_model_matrix_by_name[name], 'A1-2', animal_info.connection_matrix_vmax, animal_info.SC_focus_direction, animal_info.source_name, animal_info.target_name, 'model')
        position = animal_info.injection_site_2D['A1-2'][0]
        radius = 0.4
        cf.rm_ax_axis(axes_by_tag['e'])
        uf.plot_show_area_boundary(axes_by_tag['e'], animal=animal, dataset_key=dataset_key, show_area_name=False, edgecolor=cf.BLACK)
        cf.add_circle_patch(axes_by_tag['e'], center=position, radius=radius, edgecolor=cf.RED, facecolor=cf.RED)
        cf.rm_ax_axis(axes_by_tag['f'])
        simulation_dir = Path(model.run_dir) / 'simulation'
        source_points = np.load(simulation_dir / 'source_points.npy', mmap_mode='r', allow_pickle=False)
        target_points = np.load(simulation_dir / 'target_points.npy', mmap_mode='r', allow_pickle=False)
        plot_source_points_by_target_position(axes_by_tag['f'], source_points, target_points, position, radius, animal, dataset_key)
        cf.save_fig(fig, self.figdir + '/marmoset_area_performance_and_retrograde', **self.save_fig_kwargs)

    def get_long_range_sum(self, name):
        animal_info = self.animal_info_by_name[name]
        lambda_value = self.model_by_name[name].get_unique_Lambda()
        result = uf.get_category_longrange_connection(self.data_matrix_by_name[name], animal_info.animal, animal_info.dataset_key, longrange_threshold=lambda_value * 5, dim='2D')
        return result['longrange_connection_sum']

    def plot_long_range_connection_analysis(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 2}, get_suitable_fig_size_kwargs={'nrows': 2, 'which': 'adjust_params_custom'})
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            axes = axes_by_name[name]
            plot_removed_long_range_correlation(axes[0], animal_info.distance_2D, self.data_matrix_by_name[name], self.masked_model_matrix_by_name[name], animal_info.animal, animal_info.dataset_key, self.model_by_name[name].get_unique_Lambda(), animal_info.w)
            plot_category_long_range_connection(axes[1], self.get_long_range_sum(name))
        add_multi_panel_tags(axes_by_name, self.dataset_name_list, row_first=False)
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list, y=1.05)
        cf.save_fig(fig, self.figdir + '/long_range_connection_analysis', **self.save_fig_kwargs)

    def plot_angle_uniformity_by_distance_threshold(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 1}, get_suitable_fig_size_kwargs={'nrows': 1, 'which': 'adjust_params_custom'})
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            thresholds = np.linspace(0, np.nanmax(animal_info.distance_2D.values), 40)[:-2]
            plot_angle_uniformity_by_distance_threshold(axes_by_name[name], animal_info.distance_2D, animal_info.angle_2D, self.data_matrix_by_name[name], self.minimal_tol_by_name[name], thresholds, self.model_by_name[name].get_unique_Lambda())
        add_multi_panel_tags(axes_by_name, self.dataset_name_list, row_first=False)
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list, y=1.05)
        cf.save_fig(fig, self.figdir + '/angle_uniformity_by_distance_threshold', **self.save_fig_kwargs)

    def plot_cross_species_scaling(self):
        fig, axes = cf.get_fig_ax(ncols=3)
        axes_by_tag = {'a': axes[0], 'b': axes[1], 'c': axes[2]}
        distance_matrices = [self.animal_info_by_name[name].distance_2D for name in self.dataset_name_list]
        lambda_values = [self.model_by_name[name].get_unique_Lambda() for name in self.dataset_name_list]
        plot_scaling_law(axes_by_tag['a'], self.animal_list, distance_matrices, lambda_values)
        long_range_sums = [self.get_long_range_sum(name) for name in self.dataset_name_list]
        plot_long_range_correlation_heatmap(axes_by_tag['b'], self.animal_list, long_range_sums)
        macaque_performance = self.model_by_name['macaque_markov40'].get_area_level_correlation()
        marmoset_performance = self.model_by_name['marmoset_majka'].get_area_level_correlation()
        uf.plot_visualize_macaque_marmoset_performance_similarity(axes_by_tag['c'], macaque_performance, marmoset_performance)
        cf.add_axes_dict_tag(axes_dict=axes_by_tag, ref_ax=axes[0])
        cf.save_fig(fig, self.figdir + '/cross_species_scaling', **self.save_fig_kwargs)

    def plot_marmoset_direction_and_injection_bias(self):
        name = 'marmoset_majka'
        animal = 'marmoset'
        dataset_key = 'majka'
        animal_info = self.animal_info_by_name[name]
        model = self.model_by_name[name]
        fig, axes = cf.get_fig_ax(ncols=3)
        axes_by_tag = {'a': axes[0], 'b': axes[1], 'c': axes[2]}
        diagonal_data = animal_info.SCt_data
        diagonal_model = model.SC_model_dict[uf.SC_type_to_SC_dict_key_include_diagonal(animal_info.SC_type)]
        diagonal_performance = uf.area_level_correlation(SC_data=diagonal_data, SC_model=diagonal_model, animal=animal, dataset_key=dataset_key, diagonal=True)
        direction_preference = uf.get_direction_preference(self.data_matrix_by_name[name], animal, dataset_key, mode='uniform', SC_model=self.masked_model_matrix_by_name[name])
        plot_direction_preference_performance(axes_by_tag['a'], direction_preference, diagonal_performance)
        cf.move_ax(axes_by_tag['a'], d_right=0.06, keep_size=True)
        which_area = 'A19M'
        plot_connection_on_flatmap(axes_by_tag['b'], animal, dataset_key, self.data_matrix_by_name[name], which_area, animal_info.connection_matrix_vmax, animal_info.SC_focus_direction, animal_info.source_name, animal_info.target_name, 'data')
        cf.add_text(axes_by_tag['b'], text=which_area, x=0.5, y=-0.15, ha='center', va='center', fontsize=cf.LABEL_SIZE)
        injection_x, injection_y = animal_info.injection_site_2D[which_area][0]
        cf.plt_scatter(axes_by_tag['b'], injection_x, injection_y, color=cf.YELLOW, label='Injection site', s=50)
        cf.set_ax(axes_by_tag['b'], xlim=[injection_x - 5, injection_x + 5], ylim=[injection_y - 5, injection_y + 5])
        deviation = uf.get_injection_deviation_rate(animal=animal, dataset_key=dataset_key, mode='3D')
        plot_injection_deviation_performance(axes_by_tag['c'], deviation, diagonal_performance)
        cf.add_axes_dict_tag(axes_dict=axes_by_tag, ref_ax=axes[0])
        cf.save_fig(fig, self.figdir + '/marmoset_direction_and_injection_bias', **self.save_fig_kwargs)

    def plot_distance_2d_3d_and_edr(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 2}, get_suitable_fig_size_kwargs={'nrows': 2, 'which': 'adjust_params_custom'})
        fig.set_figwidth(fig.get_figwidth() * 0.996686)
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            axes = axes_by_name[name]
            plot_2d_3d_distance_scatter(axes[0], animal_info.distance_2D, animal_info.distance_3D)
            plot_distance_distribution(axes[1], animal_info.distance_2D, self.data_matrix_by_name[name], self.minimal_tol_by_name[name], 'data', linregress=True)
        add_multi_panel_tags(axes_by_name, self.dataset_name_list)
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list)
        cf.save_fig(fig, self.figdir + '/distance_2d_3d_and_edr', **self.save_fig_kwargs)

    def plot_angle_distance_dependence(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 2}, get_suitable_fig_size_kwargs={'nrows': 2, 'which': 'adjust_params_custom'})
        n_neighbors = 20
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            axes = axes_by_name[name]
            weight_mask = uf.get_weight_category_mask(animal_info.animal, animal_info.dataset_key, diagonal=False)['all']
            lambda_value = self.model_by_name[name].get_unique_Lambda()
            thresholds = np.linspace(0, np.nanmax(animal_info.distance_2D.values), 40)
            line_xlim = [lambda_value if name == 'human_lariviere' else 2 * lambda_value, None]
            plot_angle_distance_scatter(axes[0], weight_mask, animal_info.angle_2D, animal_info.distance_2D, n_neighbors)
            plot_angle_distance_dependence(axes[1], weight_mask, animal_info.angle_2D, animal_info.distance_2D, thresholds, n_neighbors, lambda_value, line_xlim)
            axes[1].set_xlim(axes[0].get_xlim())
        add_multi_panel_tags(axes_by_name, self.dataset_name_list)
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list, y=1.05)
        cf.save_fig(fig, self.figdir + '/angle_distance_dependence', **self.save_fig_kwargs)

    def plot_marmoset_atlas_comparison(self):
        fig, axes = cf.get_fig_ax(nrows=2, ncols=2)
        plot_data_model_connection_matrices(axes[0, 0], axes[1, 0], self.data_matrix_by_name['marmoset_majka'], self.model_matrix_by_name['marmoset_majka'], 'marmoset', 'majka')
        plot_data_model_connection_matrices(axes[0, 1], axes[1, 1], self.data_matrix_by_name['marmoset_V4'], self.model_matrix_by_name['marmoset_V4'], 'marmoset', 'V4')
        cf.add_axes_title(axes[:, 0], title='Paxinos', title_size=cf.SUP_TITLE_SIZE)
        cf.add_axes_title(axes[:, 1], title='MBM V4', title_size=cf.SUP_TITLE_SIZE)
        cf.add_axes_list_tag_by_order(axes.flatten().tolist())
        cf.save_fig(fig, self.figdir + '/marmoset_atlas_comparison', **self.save_fig_kwargs)

    def plot_connection_statistics_other_species(self):
        names = [name for name in self.dataset_name_list if name != 'marmoset_majka']
        animals = [self.animal_info_by_name[name].animal for name in names]
        margin = cf.MARGIN.copy()
        fig, subfigures, axes_by_name = get_multi_panel_figure(names, layout='row', get_ax_kwargs={'nrows': 2, 'ncols': 3, 'margin': margin}, get_suitable_fig_size_kwargs={'nrows': 2, 'ncols': 3, 'margin': margin, 'which': 'margin'})
        for name in names:
            animal_info = self.animal_info_by_name[name]
            axes = axes_by_name[name]
            plot_distance_connection_scatter(axes[0, 0], animal_info.distance_2D, self.data_matrix_by_name[name], animal_info.animal, animal_info.dataset_key, 'data')
            plot_distance_connection_scatter(axes[1, 0], animal_info.distance_2D, self.masked_model_matrix_by_name[name], animal_info.animal, animal_info.dataset_key, 'model')
            plot_connection_distribution(axes[0, 1], self.data_matrix_by_name[name], 'data')
            plot_connection_distribution(axes[1, 1], self.masked_model_matrix_by_name[name], 'model')
            plot_distance_distribution(axes[0, 2], animal_info.distance_2D, self.data_matrix_by_name[name], self.minimal_tol_by_name[name], 'data')
            plot_distance_distribution(axes[1, 2], animal_info.distance_2D, self.masked_model_matrix_by_name[name], self.minimal_tol_by_name[name], 'model')
            cf.reduce_ax_tick(axes[0, 1], axis='y', max_num=4)
            cf.reduce_ax_tick(axes[1, 1], axis='y', max_num=4)
        add_multi_panel_tags(axes_by_name, names, layout='row')
        add_multi_panel_titles(subfigures, names, animals, y=1.02)
        cf.save_fig(fig, self.figdir + '/connection_statistics_other_species', **self.save_fig_kwargs)

    def plot_area_performance_all_species(self):
        margin = cf.MARGIN.copy()
        margin['top'] = 0.8
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, layout='grid', subfig_ncols=2, subfig_nrows=2, get_ax_kwargs={'nrows': 1, 'margin': margin}, get_suitable_fig_size_kwargs={'nrows': 1, 'ax_width': cf.AX_WIDTH * 2, 'margin': margin, 'which': 'margin'})
        fig.set_figheight(fig.get_figheight() * 0.924171)
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            plot_area_level_performance(axes_by_name[name], self.data_matrix_by_name[name], self.model_matrix_by_name[name], animal_info.animal, animal_info.dataset_key)
        human_ax = axes_by_name['human_lariviere']
        ticks = human_ax.get_xticks()
        labels = human_ax.get_xticklabels()
        human_ax.set_xticks(ticks=ticks[::5], labels=[label.get_text() for label in labels[::5]], rotation=90, fontsize=cf.TICK_SIZE)
        add_multi_panel_tags(axes_by_name, self.dataset_name_list, layout='grid')
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list)
        cf.save_fig(fig, self.figdir + '/area_performance_all_species', **self.save_fig_kwargs)

    def plot_distance_relative_error(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 1}, get_suitable_fig_size_kwargs={'nrows': 1, 'which': 'adjust_params_custom'})
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            relative_error = self.model_by_name[name].get_relative_error()
            plot_distance_relative_error(axes_by_name[name], animal_info.distance_2D, relative_error, self.data_matrix_by_name[name], animal_info.source_name, animal_info.target_name)
        add_multi_panel_tags(axes_by_name, self.dataset_name_list)
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list, y=1.05)
        cf.save_fig(fig, self.figdir + '/distance_relative_error', **self.save_fig_kwargs)

    def plot_direction_preference_performance(self):
        fig, subfigures, axes_by_name = get_multi_panel_figure(self.dataset_name_list, get_ax_kwargs={'nrows': 1}, get_suitable_fig_size_kwargs={'nrows': 1, 'which': 'adjust_params_custom'})
        for name in self.dataset_name_list:
            animal_info = self.animal_info_by_name[name]
            performance = uf.area_level_correlation(SC_data=self.data_matrix_by_name[name], SC_model=self.masked_model_matrix_by_name[name], animal=animal_info.animal, dataset_key=animal_info.dataset_key, diagonal=False)
            preference = uf.get_direction_preference(self.data_matrix_by_name[name], animal_info.animal, animal_info.dataset_key, mode='uniform', SC_model=self.masked_model_matrix_by_name[name])
            plot_direction_preference_performance(axes_by_name[name], preference, performance)
        add_multi_panel_tags(axes_by_name, self.dataset_name_list)
        add_multi_panel_titles(subfigures, self.dataset_name_list, self.animal_list)
        cf.save_fig(fig, self.figdir + '/direction_preference_performance', **self.save_fig_kwargs)

    def table_1(self):
        proportions = {}
        for animal, dataset_key, name in zip(self.animal_list, self.dataset_key_list, self.dataset_name_list):
            category_proportions = uf.SC_proportion_in_weight_category(animal=animal, dataset_key=dataset_key)
            proportions[name] = {}
            for category in ['weak', 'weak_medium', 'medium_strong', 'strong']:
                proportions[name][category] = category_proportions[category] / category_proportions['all']
        cf.save_dict(proportions, self.figdir + '/table_1')

