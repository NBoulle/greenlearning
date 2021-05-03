import io
from setuptools import setup
from setuptools import find_packages

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name="GreenLearning",
    version="1.0",
    description="Deep learning library for learning Green's functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolas Boulle",
    author_email="boulle@maths.ox.ac.uk",
    url="https://github.com/NBoulle/greenlearning",
    download_url="https://github.com/NBoulle/greenlearning/tarball/v1.0",
    license="Apache-2.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "Neural Networks",
        "Scientific computing",
        "Green's functions",
        "PDE learning",
    ],
    packages=find_packages(),
    include_package_data=True,
)
