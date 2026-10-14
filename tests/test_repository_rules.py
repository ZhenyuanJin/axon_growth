import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_executable_sources_are_neutral_and_self_contained():
    forbidden = [
        re.compile(r"[\u4e00-\u9fff]"),
        re.compile(r"axon[ _-]?growth", re.IGNORECASE),
        re.compile(r"/data/zyjin/common_func"),
        re.compile(r"import common_functions"),
        re.compile(r"sys\.path\.(append|insert)"),
    ]
    failures = []
    for path in ROOT.glob("code/**/*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(source):
                failures.append(f"{path}: {pattern.pattern}")
    assert not failures, "\n".join(failures)


def test_notebooks_are_converted_to_python():
    assert not list(ROOT.glob("**/*.ipynb"))


def test_converted_workflows_do_not_use_parent_relative_paths():
    workflow_dirs = ["data_analysis", "result_analysis", "simulation", "visualization"]
    failures = []
    for directory in workflow_dirs:
        for path in (ROOT / "code" / directory).glob("*.py"):
            if "../" in path.read_text(encoding="utf-8"):
                failures.append(str(path))
    assert not failures, "\n".join(failures)


def test_plotting_does_not_restore_single_or_multi_visualizers():
    source = (ROOT / "code" / "utils_function" / "paper_figures.py").read_text(encoding="utf-8")
    assert "SingleCaseVisualizer" not in source
    assert "MultiCaseVisualizer" not in source
    assert "single_visualizer" not in source
    assert "multi_visualizer" not in source


def test_plot_functions_do_not_accept_case_containers():
    source = (ROOT / "code" / "utils_function" / "paper_figures.py").read_text(encoding="utf-8")
    assert "case_dict" not in source
    assert "case_data" not in source
    assert "getattr(" not in source
