"""Check component area fractions and preservation of area-level FLNe weights."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import shapely.geometry

from experiments.marmoset_target_conditioned_angle_edr import run_experiment as experiment


class ComponentWeightsTest(unittest.TestCase):
    def test_split_weights_and_unsplit_area_invariance(self):
        names = ["S", "U", "T"]
        info = SimpleNamespace(
            full_name=names, source_name=names,
            name_to_split_name_dict={"S": ["S--1", "S--2"], "U": ["U"], "T": ["T--1", "T--2"]},
        )
        polygons = {
            "S--1": shapely.geometry.box(0, 0, 7, 1),
            "S--2": shapely.geometry.box(10, 0, 13, 1),
            "U": shapely.geometry.box(20, 0, 120, 10),
            "T--1": shapely.geometry.box(0, 20, 2, 21),
            "T--2": shapely.geometry.box(10, 20, 18, 21),
        }
        SC_data = pd.DataFrame({"T": [0.6, 0.4, 0]}, index=names)
        components = experiment.get_area_components(info, polygons)
        _, weights = experiment.get_angle_centers_and_weights("T", info, SC_data, components)
        np.testing.assert_allclose(weights, [0.084, 0.336, 0.036, 0.144, 0.08, 0.32])
        np.testing.assert_allclose(weights[:4].sum(), 0.6)
        np.testing.assert_allclose(weights[4:].sum(), 0.4)
        np.testing.assert_array_equal(components["U"][1], [1.0])

        polygons["U"] = shapely.geometry.box(20, 0, 1020, 100)
        components = experiment.get_area_components(info, polygons)
        _, rescaled_weights = experiment.get_angle_centers_and_weights("T", info, SC_data, components)
        np.testing.assert_array_equal(rescaled_weights, weights)

        info.name_to_split_name_dict["T"] = ["T"]
        polygons["T"] = shapely.geometry.box(0, 20, 50, 25)
        components = experiment.get_area_components(info, polygons)
        _, weights = experiment.get_angle_centers_and_weights("T", info, SC_data, components)
        np.testing.assert_allclose(weights, [0.42, 0.18, 0.4])


if __name__ == "__main__":
    unittest.main()
