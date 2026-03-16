try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as f:
    readme = f.read()

setup(
    name='tensorzinb',
    version='0.0.2',
    description='Zero Inflated Negative Binomial Model for Single-cell RNA-Sequencing Analysis using TensorFlow',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Tao Cui',
    author_email='taocui.caltech@gmail.com',
    url='https://github.com/wanglab/tensorzinb',
    keywords='Zero Inflated Negative Binomial scRNA-seq',
    packages=['tensorzinb'],
    include_package_data=True,
    python_requires='>=3.9,<3.13',
    install_requires=[
        'tensorflow>=2.16',
        'tf-keras>=2.16',
        'numpy>=1.23.5',
        'pandas',
        'patsy',
        'scikit_learn',
        'scipy',
        'statsmodels',
    ],
    license='Apache',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ]
)
