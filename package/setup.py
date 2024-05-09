from setuptools import setup, find_packages

requirements = ['librosa==0.10.1',
'matplotlib==3.7.2',
'numpy==1.24.3',
'pandas==1.5.3',
'PyYAML==6.0.1',
'PyYAML==6.0.1',
'scikit_learn==1.3.2',
'torch==2.2.1',
'torchaudio==2.2.1',
'tqdm==4.65.0',
'transformers==4.32.1',  
    'gdown'
]

setup(name='caser',
      version='0.1',
      author='Bashar M. Deeb',
      author_email='Bashar.betta9@gmail.com',
      packages=find_packages('.'),
      url='https://github.com/BasharBetta7/SER/tree/master',
      description='CA-SER python library for speech emotion recognition',
      keywords='audio speech emotion recognition',
      install_requires=requirements,
      )
