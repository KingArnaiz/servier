from setuptools import find_packages, setup

__version__ = "1.5.2"

# Load README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='servier',
    author='Ronrick Daano',
    author_email='ronrickarnaiz@gmail.com',
    description='Servier Data Science Technical Test',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/KingArnaiz/servier',
    download_url=f'https://github.com/KingArnaiz/servier/v_{__version__}.tar.gz',
    packages=find_packages(),
    package_data={'servier': ['py.typed']},
    entry_points={
        'console_scripts': [
            'servier_train=servier.train:chemprop_train',
            'servier_predict=servier.train:chemprop_predict',
            'servier_fingerprint=servier.train:chemprop_fingerprint',
            'servier_hyperopt=servier.hyperparameter_optimization:chemprop_hyperopt',
            'servier_interpret=servier.interpret:chemprop_interpret',
            'servier_web=servier.web.run:chemprop_web',
            'sklearn_train=servier.sklearn_train:sklearn_train',
            'sklearn_predict=servier.sklearn_predict:sklearn_predict',
        ]
    },
    install_requires=[
        'flask>=1.1.2',
        'hyperopt>=0.2.3',
        'matplotlib>=3.1.3',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'pandas-flavor>=0.2.0',
        'scikit-learn>=0.22.2.post1',
        'scipy>=1.5.2',
        'sphinx>=3.1.2',
        'tensorboardX>=2.0',
        'torch>=1.5.2',
        'tqdm>=4.45.0',
        'typed-argument-parser>=1.6.1'
    ],
    extras_require={
        'test': [
            'pytest>=6.2.2',
            'parameterized>=0.8.1'
        ]
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords=[
        'chemistry',
        'machine learning',
        'property prediction',
        'message passing neural network',
        'graph neural network'
    ]
)
