PYTHON ?= python

inplace:
	$(PYTHON) setup.py develop

all: inplace

clean:
	$(PYTHON) setup.py clean

test: inplace
	nosetests --with-coverage --cover-package=machinedesign -v
