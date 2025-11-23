from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='adore',
    version='0.1.2',
    author='Lemen Chao, Anran Fang, Ming Lei',
    author_email='chaolemen@ruc.edu.cn',
    description='Adaptive Derivative Order Randomized Explanation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/LemenChao/adore.git',
    license='GPL-3.0',
    packages=find_packages(exclude=['tests','images','data','examples','en_core_web_sm-3.8.0-py3-none-any.whl']),
    python_requires='>=3.9,<3.13',
    install_requires=[
        'numpy~=1.26.4',
        'scipy~=1.11.4',
        'joblib~=1.4.2',
        'scikit-learn~=1.5.2',
        'matplotlib~=3.9.4',
        'seaborn~=0.13.2',
        'pandas~=2.2.3',
        'opencv-python~=4.10.0.84',
        'spacy~=3.8.0',
        'nltk~=3.9.1',
        'keybert~=0.8.5',
        'tensorflow~=2.18.0',
        'tf-keras~=2.18.0',
        'torch~=2.5.1',
        'transformers~=4.46.3',
        'pyspellchecker~=0.7.2',
        'tensorflow-datasets~=4.9',
    ],
    extras_require={
        'dev': ['pytest', 'flake8'],
    },
    include_package_data=True,
    zip_safe=False,
)
