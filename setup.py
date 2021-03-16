from __future__ import (
    print_function,
    print_function
)

import os
import platform
import sys
from distutils.core import setup

from setuptools import find_packages

if sys.version_info < (3,):
    print("Python 2 has reached end-of-life and is no longer supported by H0.")
    sys.exit(-1)
if sys.platform == 'win32' and sys.maxsize.bit_length() == 31:
    print("32-bit Windows Python runtime is not supported. Please switch to 64-bit Python.")
    sys.exit(-1)

python_min_version = (3, 6, 2)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(platform.python_version(), python_min_version_str))
    sys.exit(-1)

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    version_range_max = max(sys.version_info[1], 8) + 1
    setup(
        name=os.getenv('PYHZ_PACKAGE_NAME', 'pyhashzero'),
        version='1.0.0rc0',
        description="General Purpose Python Library",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        install_requires=[
            'pytest',
            'opencv-python',
            'numpy',
            'pygame',
            'pymunk',
            'torch',
            'six',
            'gym'
        ],
        package_data={},
        url='https://github.com/pyhashzero/pyhashzero',
        download_url='https://github.com/pyhashzero/pyhashzero/tags',
        author='Hamitcan Malkoç',
        author_email='hamitcanmalkoc@gmail.com',
        python_requires='>={}'.format(python_min_version_str),
        classifiers=[
                        'Development Status :: 5 - Production/Stable',
                        'Intended Audience :: Developers',
                        'Intended Audience :: Education',
                        'Intended Audience :: Science/Research',
                        'License :: OSI Approved :: BSD License',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Scientific/Engineering :: Mathematics',
                        'Topic :: Scientific/Engineering :: Artificial Intelligence',
                        'Topic :: Software Development',
                        'Topic :: Software Development :: Libraries',
                        'Topic :: Software Development :: Libraries :: Python Modules',
                        'Programming Language :: C++',
                        'Programming Language :: Python :: 3',
                    ] + ['Programming Language :: Python :: 3.{}'.format(i) for i in range(python_min_version[1], version_range_max)],
        license='MIT',
        keywords='pyhashzero machine-learning intelligent-virtual-assistant',
    )
