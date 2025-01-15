.PHONY: build publish

build:
	uv build

publish: build
	twine upload dist/*
