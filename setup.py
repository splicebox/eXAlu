import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eXAlu",
    version="0.3",
    author="Zitong He & Liliana Florea",
    author_email="hezt@jhu.edu",
    description="Deep learning based Alu exonization event detecter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    project_urls={
        "Bug Tracker": "https://github.com/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
)