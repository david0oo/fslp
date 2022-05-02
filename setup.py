from setuptools import setup, find_packages

setup(
    name='fslp',
    version='0.0.1',
    description='A prototypical python implementation of FSLP',
    author='David Kiessling',
    author_email='david.kiessling@kuleuven.be',
    url='https://github.com/david0oo/fslp',
    license='BSD',
    packages=find_packages(['fslp', 'fslp.*']),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        # 'casadi'
    ],
    keywords=['fslp']
)
