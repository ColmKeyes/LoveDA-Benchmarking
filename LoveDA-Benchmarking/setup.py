from setuptools import setup, find_packages

setup(
    name="lovebench",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=1.12",
        "torchvision>=0.13",
        "torchgeo>=0.4",
        "pytorch_lightning>=2.0",
        "numpy>=1.21",
        "tqdm>=4.64"
    ],
    entry_points={
        "console_scripts": [
            "lovebench=lovebench.cli:main",
        ],
    },
)
