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
	pip install uv
	uv pip install -e ".[dev,lint,modeling,data,pipeline]"

# lint, format, and check all files
lint:
	pre-commit run --all-files

###############################################################################
# Release and versioning

# tag a new version
tag-for-release version:
	git tag -a "{{version}}" -m "{{version}}"
	echo "Tagged: $(git tag --sort=-version:refname| head -n 1)"

# release a new version
release:
	git push --follow-tags

###############################################################################
# Infra / Data Storage

# Default region for infrastructures
default_region := "us-central1"
key_guid := replace_regex(uuid(), "([a-z0-9]{8})(.*)", "$1")
default_key := justfile_directory() + "/.keys/dev.json"
default_project := "sci-software-graph"
default_bucket := "gs://sci-software-graph-data-store"

# run gcloud login
login:
  gcloud auth login
  gcloud auth application-default login

# generate a service account JSON
gen-key project=default_project:
	mkdir -p {{justfile_directory()}}/.keys/
	gcloud iam service-accounts create {{project}}-{{key_guid}} \
		--description="Dev Service Account {{key_guid}}" \
		--display-name="{{project}}-{{key_guid}}"
	gcloud projects add-iam-policy-binding {{project}} \
		--member="serviceAccount:{{project}}-{{key_guid}}@{{project}}.iam.gserviceaccount.com" \
		--role="roles/owner"
	gcloud iam service-accounts keys create {{justfile_directory()}}/.keys/{{project}}-{{key_guid}}.json \
		--iam-account "{{project}}-{{key_guid}}@{{project}}.iam.gserviceaccount.com"
	cp -rf {{justfile_directory()}}/.keys/{{project}}-{{key_guid}}.json {{default_key}}
	@ echo "----------------------------------------------------------------------------"
	@ echo "Sleeping for thirty seconds while resources set up"
	@ echo "----------------------------------------------------------------------------"
	sleep 30
	@ echo "Be sure to update the GOOGLE_APPLICATION_CREDENTIALS environment variable."
	@ echo "----------------------------------------------------------------------------"

# create a new gcloud project and generate a key
init project=default_project:
	gcloud projects create {{project}} --set-as-default
	echo "----------------------------------------------------------------------------"
	echo "Follow the link to setup billing for the created GCloud account."
	echo "https://console.cloud.google.com/billing/linkedaccount?project={{project}}"
	echo "----------------------------------------------------------------------------"
	just gen-key {{project}}

# switch active gcloud project
switch-project project=default_project:
	gcloud config set project {{project}}

# enable gcloud services
enable-services project=default_project:
	gcloud services enable cloudresourcemanager.googleapis.com
	gcloud services enable \
		storage.googleapis.com \
		compute.googleapis.com \
		logging.googleapis.com \
		monitoring.googleapis.com

# setup all resources
setup-infra project=default_project:
	just enable-services {{project}}
	-\gcloud storage buckets create {{default_bucket}}
	gcloud storage buckets add-iam-policy-binding {{default_bucket}} --member=allUsers --role=roles/storage.objectViewer

###############################################################################
# Database Management

db_path := justfile_directory() + "/rs_graph/data/files/rs-graph"
current_git_details := `git log -n 1 --pretty=format:"Git Commit: %h"`

# create a new migration (can only run if git is clean)
db-migrate target="dev":
	git update-index --really-refresh
	sed -i "s|REPLACE_SQLITE_DB_PATH|{{db_path}}-{{target}}.db|g" {{justfile_directory()}}/alembic.ini
	mkdir -p {{justfile_directory()}}/rs_graph/data/files
	-alembic revision --autogenerate -m "{{current_git_details}}"
	-alembic upgrade head
	sed -i "s|{{db_path}}-{{target}}.db|REPLACE_SQLITE_DB_PATH|g" {{justfile_directory()}}/alembic.ini
	sleep 0.5
	git add -A
	git commit -m "New migration for target: {{target}}"

db-upgrade target="dev":
	sed -i "s|REPLACE_SQLITE_DB_PATH|{{db_path}}-{{target}}.db|g" {{justfile_directory()}}/alembic.ini
	mkdir -p {{justfile_directory()}}/rs_graph/data/files
	-alembic upgrade head
	sed -i "s|{{db_path}}-{{target}}.db|REPLACE_SQLITE_DB_PATH|g" {{justfile_directory()}}/alembic.ini

###############################################################################
# Docker management

# build docker image locally
docker-build:
	docker build --tag rs-graph {{justfile_directory()}}

# run docker in interactive mode with bash
docker-run:
	docker run --rm -it rs-graph bash

###############################################################################
# Publications management

# hot-reload quarto doc
quarto-serve project="qss-code-authors":
	-quarto preview {{justfile_directory()}}/publications/{{project}}/qss-code-authors.qmd

# render quarto doc
quarto-render project="qss-code-authors":
	-quarto render {{justfile_directory()}}/publications/{{project}}/qss-code-authors.qmd