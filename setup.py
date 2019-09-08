import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="module_ai_pkg",
    version="0.0.1",
    author="Pascal Sigel",
    author_email="pascal.sige@innovacionyrobotica.com",
    description="Module to dive in ASTNG data",
    long_description="Module to dive in ASTNG data",
    long_description_content_type="text/markdown",
    url="https://github.com/astng/ai_data_analysis",
    packages=setuptools.find_packages(),
    license="Proprietary",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
