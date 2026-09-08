install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

recommend:
	python main.py recommend --title "Avatar" --top-k 10

evaluate:
	python main.py evaluate --sample-size 300 --top-k 10

analyze:
	python main.py analyze
