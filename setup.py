from setuptools import setup, find_packages

setup(
    name='xia_eeg_toolkit',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/dddd1007/xia_eeg_toolkit',
    author='Xia Xiaokai',
    author_email='xia@xiaokai.me',
    description='My workflow of EEG data analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'mne',
        'toml',
        'matplotlib',
        'multiprocessing'
    ],
)