from setuptools import setup, find_packages

NAME = "cv-utils"
VERSION = "0.0.2"
DESCRIPTION = "Package to help developers with their opencv-python adventures."
AUTHOR = "Jessy van Polanen"
AUTHOR_EMAIL = "jessyvanpolanen@gmail.com"
KEYWORDS = ["cv-utils", "opencv", "computer vision", "utils", "python"]
LICENSE = "MIT"
URL = "https://github.com/JVP11292021/CV-Utils.git"


def read_me():
    with open("README.md") as file:
        return file


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=read_me(),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    classifiers=[
        "Development status :: 0.0.2 - Alpha",
        "Intended audience :: Developers",
        "Operating system :: MacOS :: MacOS X",
        "Operating system :: microsoft :: Windows",
        "Programming language :: Python :: 3",
        "Programming language :: Python :: 3.6",
        "Programming language :: Python :: 3.7",
        "Programming language :: Python :: 3.8",
        "Programming language :: Python :: 3.9",
    ],
    keyword=KEYWORDS,
    packages=find_packages(exclude=["docs", "tests*", "contrib"]),
    install_requires=[
        "numpy",
        "opencv-python",
        "cmake",
        "face_recognition",
        "mediapipe",
        "pillow",
        "numba"
    ],
)
