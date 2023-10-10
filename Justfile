# list all available commands
default:
  just --list

###############################################################################
# Basic project and env management

# clean all build, python, and lint files
clean:
	rm -fr dist
	rm -fr .eggs
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .mypy_cache

# install with all deps
install:
    pip install -e ".[dev,lint]"

# lint, format, and check all files
lint:
	pre-commit run --all-files

###############################################################################
# Quarto

# store various dirs and filepaths
NOTEBOOKS_DIR := justfile_directory() + "/notebooks/"
FILE_URI := NOTEBOOKS_DIR + "viz.ipynb"
BUILD_DIR := NOTEBOOKS_DIR + "_build/"

# install with quarto deps
quarto-install env_name="rs-graph":
	just install
	conda install -y -c conda-forge \
		"r-base=4.3.1" \
		"r-essentials" \
		"r-jsonlite" \
		"r-tinytex" \
		"texlive-core"

# remove build files
quarto-clean:
	rm -fr {{BUILD_DIR}}
	rm -fr {{NOTEBOOKS_DIR}}/.quarto

# watch file, build, and serve
quarto-watch:
	quarto preview {{FILE_URI}} --to html

# build page
quarto-build:
	quarto render {{FILE_URI}} --to html
	touch {{BUILD_DIR}}.nojekyll

###############################################################################
# Release and versioning

# tag a new version
tag-for-release version:
	git tag -a "{{version}}" -m "{{version}}"
	echo "Tagged: $(git tag --sort=-version:refname| head -n 1)"

# release a new version
release:
	git push --follow-tags