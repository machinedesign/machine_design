PYTHON ?= python
PYTEST ?= pytest

inplace:
	$(PYTHON) setup.py develop

all: inplace

clean:
	$(PYTHON) setup.py clean

test: inplace
	$(PYTEST) --cov=machinedesign --cov-report term-missing -v
