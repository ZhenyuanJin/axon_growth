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
