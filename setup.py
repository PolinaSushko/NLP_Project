from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path : str) -> List[str]:
    """
    Reads a requirements file and returns a list of package requirements.

    Args:
        file_path (str): The path to the requirements file.

    Returns:
        List[str]: A list of strings, each representing a package requirement without newline characters.
    """

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'NLP_project',
    version = '0.0.1',
    author = 'Polina',
    author_email = 'polss2004@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)