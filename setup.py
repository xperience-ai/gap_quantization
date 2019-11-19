#pylint:disable=missing-docstring
import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open('requirements.txt', 'r') as req_file:
    INSTALL_REQS = [line.strip() for line in req_file.readlines()]

setuptools.setup(
    name='gap_quantization',
    version='0.1',
    author="Maxim Zemlyanikin",
    author_email="maxim.zemlyanikin@xperience.ai",
    description="Set of utilities to quantize and export PyTorch models for GreenWaves GAP8 chip",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/xperience-ai/gap_quantization",
    license="BSD",
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQS)
