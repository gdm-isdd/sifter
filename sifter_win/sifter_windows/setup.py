from setuptools import setup, find_packages

setup(
    name="sifter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rdkit-pypi",
        "biopython",
        "pyqt5",
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "sifter=sifter.sifter:main"
        ]
    },
    author="Gabriele De Marco",
    description="A tool to run SIFt analysis and optionally visualize pharmacophores.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gdm-isdd/sifter",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
)
