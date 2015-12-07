from distutils.core import setup

requirements = ['pandas>=0.17.0']

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='RecEval',
    description='A light-weight recommendation systems evaluation framework.',
    long_description=readme,
    author='Ahmed Kachkach',
    author_email='ahmed.kachkach@gmail.com',
    version='0.1',
    packages=['receval'],
    install_requires=requirements,
    license='Apache 2.0'
)
