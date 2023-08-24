from setuptools import setup, find_packages

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

with open("tests/requirements.txt", "r") as fh:
    requirements_tests = fh.readlines()

setup(
    name="ringtransformer",
    version="0.0.1",
    description="Ring Transformer for Point Cloud Segmentation",
    url="https://shash",
    license='IP of Shahsitha Kularatna',
    author="Shashitha Kularatna",
    author_email="shash.kularatna@correvate.co.uk",
    install_requires=requirements,
    tests_requires=requirements_tests,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.8.*'
)
