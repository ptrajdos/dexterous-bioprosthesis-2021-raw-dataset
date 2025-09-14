from setuptools import setup, find_packages
import platform

def numpy_version():
    if "armv" in platform.machine():
        return 'numpy>=1.22.4'
    return 'numpy>=1.22.4'

def numba_version():
    if "armv" in platform.machine():
        #The highest version that uses LLVM 11 or 13
        #Newer versions use LLVM 14 unavailable on Raspberry
        return 'numba==0.56.4' 

    return 'numba'

def scikit_learn_version():
    if "armv" in platform.machine():
        return "scikit-learn>=1.2.2"

    return "scikit-learn>=1.2.2"


setup(
        name='dexterous_bioprosthesis_2021_raw_datasets',
        version ='0.0.5',
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
                'PyWavelets>=1.4.1'
                
        ],
        test_suite='test'
        )
