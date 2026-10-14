PYTHON ?= python
export PYTHONPATH := $(CURDIR)/code$(if $(PYTHONPATH),:$(PYTHONPATH))
.DEFAULT_GOAL := help

.PHONY: help connection-generation parameter-search result-analysis paper-figures test

help:
	@echo "Available targets:"
	@echo "  connection-generation  Generate primary connection-model results"
	@echo "  parameter-search       Run the primary parameter search"
	@echo "  result-analysis        Summarize parameter-search results"
	@echo "  paper-figures          Generate all main and supplementary figures"
	@echo "  test                   Run the repository test suite"

connection-generation:
	$(PYTHON) code/simulation/run_connection_generation.py

parameter-search:
	$(PYTHON) code/simulation/run_parameter_search.py

result-analysis:
	$(PYTHON) code/result_analysis/run_result_analysis.py

paper-figures:
	$(PYTHON) code/visualization/run_paper_figures.py

test:
	$(PYTHON) -m pytest -q
