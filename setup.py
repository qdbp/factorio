from setuptools import setup

# semver with automatic minor bumps keyed to unix time
__version__ = "0.2.0"

setup(
    python_requires=">=3.7.0",
    install_requires=[
        'SLPP-23',
        'pulp',
        'numpy',
        'networkx',
        'matplotlib',
        'pygraphviz',
    ],
    extras_require={
        'test': [
            'pytest',
        ],
    },
    name="factorio",
    packages=["factorio"],
    entry_points={
        'console_scripts': [
            'balancer_solver = factorio.balancer_solver:main',
            'belt_solver = factorio.belt_solver:main',
        ],
    },
    version=__version__,
)
