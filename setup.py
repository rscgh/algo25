from setuptools import setup, find_packages

setup(
    name='brainannlib',  # Name of the package
    version='0.1',
    packages=find_packages(),
    #packages=find_packages(where='lib'),
    #package_dir={'': 'lib'},  # Map 'lib' directory to the package    
    pipinstall_requires=[],  # Any dependencies your package needs
)