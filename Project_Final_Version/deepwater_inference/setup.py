import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepwater-xlux",
    version="0.0.1",
    author="Filip Lux",
    author_email="xlux@fi.muni.cz",
    description="deepwater",
    long_description='cell segmentation tool',
    long_description_content_type="text/markdown",
    url="https://github.com/xlux/deepwater",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
