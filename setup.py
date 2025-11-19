from pathlib import Path
import re
from setuptools import setup, find_packages
import platform
import sys

def numpy_version():
    if "armv" in platform.machine():
        return 'numpy>=1.22.4'
    return 'numpy>=1.22.4'

def numba_version():
    if "armv" in platform.machine():
        #The highest version that uses LLVM 11 or 13
        #Newer versions use LLVM 14 unavailable on Raspberry
        if sys.version_info > (3,10) and sys.version_info < (3,12):
            return 'numba==0.58.0' 
        if sys.version_info >= (3,9) and sys.version_info < (3,10):
            return 'numba==0.56.4'

    return 'numba'

def scikit_learn_version():
    if "armv" in platform.machine():
        return "scikit-learn>=1.2.2"

    return "scikit-learn>=1.2.2"

def get_version():
    init_py = Path(__file__).parent / "dexterous_bioprosthesis_2021_raw_datasets" / "__init__.py"
    text = init_py.read_text(encoding='utf-8')
    version_pattern = '__version__\s*=\s*[\'"]([^\'"]+)[\'"]'
    version_match = re.search(
        version_pattern,
        text
    )
    if not version_match:
        raise RuntimeError("Unable to find version string.")
    version_found = version_match.group(1)
    return version_found

setup(
        name='dexterous_bioprosthesis_2021_raw_datasets',
        version =get_version(),
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/dexterous-bioprosthesis-2021-raw-dataset',
        description="Dataset creation framework. Instances are represented by raw signals objects",
        packages=find_packages(include=[
                'dexterous_bioprosthesis_2021_raw_datasets',
                'dexterous_bioprosthesis_2021_raw_datasets.*',
                ]),
        install_requires=[ 
                'pandas<3.0.0',
                numpy_version(),
                'matplotlib',
                'scipy>=1.12.0',
                'liac-arff',
                'joblib',
                scikit_learn_version(),
                'tqdm',
                'joblib',
                'dtw-python',
                'Cython',
                'fastdtw',
                'pygad>=2.18.3, <3.0.0',
                'kneed',
                'librosa',
                'audiomentations', 
                numba_version(),
                'statsmodels>=0.13.5',
                'PyWavelets>=1.4.1',
                'hyperopt @ git+https://github.com/hyperopt/hyperopt.git',
                'sktime>=0.38.5',
                
        ],
        test_suite='test'
        )
