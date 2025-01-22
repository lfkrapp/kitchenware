.PHONY: build publish

build:
	uv build

publish: build
	twine upload dist/*

init:
	mkinit kitchenware --relative --nomods > kitchenware/__init__.py

format:
	ruff format kitchenware
