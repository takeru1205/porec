.PHONY:\
	format\
	test\
	lint\
	
all: format lint test

format:
	poetry run isort ./porec ./tests
	poetry run black ./porec ./tests

test:
	poetry run pytest tests

lint:
	poetry run mypy ./porec ./tests
	poetry run flake8 ./porec ./tests
