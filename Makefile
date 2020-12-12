install:
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt
	python -m pip install -e .
