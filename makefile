# makefile

install:
	pip3 install --upgrade pip &&\
		pip3 install -r requirements/requirements.txt &&\
		pip3 install -r /home/akhil/Desktop/ast/ml-ops-deployment/requirements/test_requirements.txt

test:
	pytest tests/test_*.py

all: install test
