import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='gap_quantization',
     version='0.1',
     author="Maxim Zemlyanikin",
     author_email="maxim.zemlyanikin@xperience.ai",
     description="Set of utilities to quantize and export PyTorch models for GreenWaves GAP8 chip",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/xperience-ai/gap_quantization",
     license="BSD",
     packages=setuptools.find_packages(),
)