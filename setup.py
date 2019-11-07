# setup.py, setup.cfg and MANIFEST.in work together when creating distributions
# using `python setup.py sdist bdist_wheel`
import os
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

# We can import the package before the build
# because we don't need to compile anything
os.environ["RELAX_IMPORTS"] = "true"
import gym_fin

# Todo: research how to maintain multiple related libraries easily
# https://stackoverflow.com/questions/54265944/python-multiple-packages-in-one-repository-or-one-package-per-repository

version = gym_fin.__version__

setuptools.setup(
    name="gym_fin",
    version=version,
    description="Financial Services Environments for OpenAI Gym",
    long_description=long_description,
    # long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["docs", "tests"]),
    # package_data={'mystuff': ['my_data_file.dat']}, # available at runtime
    # entry_points={
    #     'console_scripts': [  # creates a CLI command
    #         'hello=example_package.myclitool:say_hello'
    #     ]
    # },
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=["gym", "numpy", "scipy"],
    url="https://github.com/schuderer/bprl",
    author="Andreas Schuderer",
    author_email="pypi@schuderer.net",
    license="MIT",
    classifiers=[  # https://pypi.org/classifiers/
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
