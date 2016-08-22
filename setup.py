from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='graphgen',
    version='0.1',
    description='Makes graphs with community structure',
    author='Nathaniel Rodriguez',
    packages=['graphgen'],
    url='https://github.com/Nathaniel-Rodriguez/graphgen.git',
    install_requires=[
          'networkx',
          'numpy'
      ],
    include_package_data=True,
    zip_safe=False)