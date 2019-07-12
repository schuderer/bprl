#!/bin/bash
set -e
# source: https://www.youtube.com/watch?v=P3dY3uDmnkU

export package=gym_fin

# Install package(s) as if they were installed using pip in local environment,
# while still keeping them editable at the source location.
pipenv install --editable .

# Run tests (TODO: more tests, coverage, CI?)
pytest $package

# Disable useless setuptools-related warning
export PYTHONDONTWRITEBYTECODE=

# Clean previous build data
# Common pitfalls:
# https://blog.ionelmc.ro/2014/06/25/python-packaging-pitfalls/#id18
rm -rf dist build */*.egg-info *.egg-info

# Build source (sdist) and binary (bdist_wheel) distribution
python setup.py sdist bdist_wheel

# Upload to test repository (need to create account first)
twine upload --repository testpypi dist/*

# Try to install from test repository
pipenv install --index-url https://test.pypi.org/simple/ $package

# Upload to pypi repository (account needed there, too)
# twine upload --repository pypi dist/*

# Install and profit!!
# pipenv install $package
