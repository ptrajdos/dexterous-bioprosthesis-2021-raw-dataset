ROOTDIR=$(realpath $(dir $(firstword $(MAKEFILE_LIST))))

PROJECTREQFILE=${ROOTDIR}/requirements.txt

SRCDIR=${ROOTDIR}/dexterous_bioprosthesis_2021_raw_datasets
TESTDIR?=${ROOTDIR}/tests
COVDIR=${ROOTDIR}/htmlcov_p
COVERAGERC=${ROOTDIR}/.coveragerc
REQ_FILE=${ROOTDIR}/requirements_dev.txt
INSTALL_LOG_FILE=${ROOTDIR}/install.log
VENV_SUBDIR=${ROOTDIR}/venv
COVERAGERC=${ROOTDIR}/.coveragerc
DOCS_DIR=${ROOTDIR}/docs
DATADIR=${ROOTDIR}/data
STATICDIR=${ROOTDIR}/static_analysis
LINTFILE=${STATICDIR}/lint.json
FLAKE8FILE=${STATICDIR}/flake8.log
MYPYFILE=${STATICDIR}/mypy.log
TESTSETNAME=Andrzej_19_10_2022
TESTDATADIR=${DATADIR}/${TESTSETNAME}
TESTDATAZIP=${DATADIR}/${TESTSETNAME}.zip

COVERAGE = coverage
UNITTEST_PARALLEL = unittest-parallel
PDOC= pdoc3
PYLINT= pylint
FLAKE8= flake8
MYPY= mypy
PYTHON=python
SYSPYTHON=python
#--system-site-packages
VENV_OPTIONS=
PIP=pip
PYTEST=pytest

LOGDIR=${ROOTDIR}/testlogs
LOGFILE=${LOGDIR}/`date +'%y-%m-%d_%H-%M-%S'`.log

PYTHON_VERSION=3.9

ifeq ($(OS),Windows_NT)
	ACTIVATE:=. ${VENV_SUBDIR}/Scripts/activate
else
	ACTIVATE:=. ${VENV_SUBDIR}/bin/activate
endif

.PHONY: all clean test docs

clean:
	rm -rf ${VENV_SUBDIR}

venv:
	${SYSPYTHON} -m venv --upgrade-deps ${VENV_OPTIONS} ${VENV_SUBDIR}
	${ACTIVATE}; ${PYTHON} -m ${PIP} install -e ${ROOTDIR} --prefer-binary --log ${INSTALL_LOG_FILE} -r ${REQ_FILE}

test: venv data_unp

	mkdir -p ${LOGDIR}
	${ACTIVATE}; ${COVERAGE} run --branch  --source=${SRCDIR} -m unittest discover -p '*_test.py' -v -s ${TESTDIR} 2>&1 |tee -a ${LOGFILE}
	${ACTIVATE}; ${COVERAGE} html --show-contexts

test_parallel: venv data_unp

	mkdir -p ${COVDIR}  ${LOGDIR}
	${ACTIVATE}; ${UNITTEST_PARALLEL} --class-fixtures -v -t ${ROOTDIR} -s ${TESTDIR} -p '*_test.py' --coverage --coverage-rcfile ./.coveragerc --coverage-source ${SRCDIR} --coverage-html ${COVDIR} 2>&1 |tee -a ${LOGFILE}

docs:
	${ACTIVATE}; $(PDOC) --force --html ${SRCDIR} --output-dir ${DOCS_DIR}

profile: venv data_unp

	${ACTIVATE}; ${PYTEST} -n auto --cov-report=html --cov=${SRCDIR} --profile ${TESTDIR}

${STATICDIR}:
	mkdir -p ${STATICDIR}
flake8: venv ${STATICDIR}
	${ACTIVATE}; ${FLAKE8} --jobs auto ${SRCDIR} > ${FLAKE8FILE} || true

mypy: venv ${STATICDIR}
	${ACTIVATE}; ${MYPY} --pretty --show-error-context ${SRCDIR} > ${MYPYFILE} || true
lint: venv ${STATICDIR}
	${ACTIVATE}; ${PYLINT} -j 0 ${SRCDIR} --output-format=json > ${LINTFILE} || true

static_check: flake8 mypy lint

${TESTDATADIR}:
	@echo "Unpacking ${TESTDATAZIP}"
	unzip ${TESTDATAZIP} -d ${DATADIR}

data_unp: ${TESTDATADIR}
	@echo "Creating data"
