# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* biodiversipy/*.py

black:
	@black scripts/* biodiversipy/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr biodiversipy-*.dist-info
	@rm -fr biodiversipy.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GCP SETUP
# ----------------------------------

### Variables
GCP_PROJECT_ID=le-wagon-bootcamp-346910
GCP_BUCKET_NAME= wagon-data-871-biodiversipy
GCP_BUCKET_TRAINING_FOLDER = 'trainings'
GCP_REGION=europe-west1
GCP_BUCKET_FOLDER=data
LOCAL_DATA_PATH = 'raw_data'
GCP_BUCKET_FILE_NAME=$(shell basename ${LOCAL_DATA_PATH})

upload_all_data:
	@gsutil cp -r ${LOCAL_DATA_PATH} gs://${GCP_BUCKET_NAME}/${GCP_BUCKET_FOLDER}/${GCP_BUCKET_FILE_NAME}

upload_data:
	@test $(path)
	@gsutil cp -r $(path) gs://${GCP_BUCKET_NAME}/${GCP_BUCKET_FOLDER}/$(path)

download_data:
	@test $(path)
	@gsutil cp -r gs://${GCP_BUCKET_NAME}/${GCP_BUCKET_FOLDER}/$(path) $(path)

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

REGION=europe-west1

PYTHON_VERSION=3.8
FRAMEWORK=TensorFlow
RUNTIME_VERSION=2.7

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=biodiversipy
FILENAME=trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=complex_taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
    --scale-tier CUSTOM \
    --master-machine-type n1-standard-16

# ----------------------------------
#      MODEL
# ----------------------------------
run_locally:
	@python biodiversipy/main.py

soilgrid_download:
	@python scripts/soilgrid_download.py

# ----------------------------------
#      STREAMLIT
# ----------------------------------

streamlit:
	-@streamlit run website/app.py

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

# heroku_login:
# 	-@heroku login

# heroku_create_app:
# 	-@heroku create ${APP_NAME}

# deploy_heroku:
# 	-@git push heroku master
# 	-@heroku ps:scale web=1
