"""Setup script for UDiff package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="udiff",
    version="1.0.0",
    description=(
        "Deep Unfolding of Diffusion Probabilistic Models "
        "for Efficient Sparse Signal Recovery"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/udiff",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="diffusion, deep unfolding, compressed sensing, sparse recovery, MRI",
)
