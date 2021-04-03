import os

from setuptools import setup, find_packages

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'rlutils', 'VERSION.txt'), 'r') as f:
    version = f.read()

with open(os.path.join(dir_path, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rlutils-python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    url='https://github.com/vermouth1992/rlutils',
    license='Apache 2.0',
    author='Chi Zhang',
    author_email='czhangseu@gmail.com',
    description='Reinforcement Learning Utilities',
    entry_points={
        'console_scripts': [
            'rlplot=rlutils.plot:main',
            'rlrun=rlutils.run:main'
        ]
    }
)
