.PHONY: build publish

init:
	mkinit kitchenware --relative --nomods > kitchenware/__init__.py

format:
	ruff format kitchenware

build:
	uv build

publish: build
	twine upload dist/*
