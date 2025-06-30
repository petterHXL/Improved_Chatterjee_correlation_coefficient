from setuptools import setup, find_packages

setup(
    name="Improved_Chatterjee_correlation_coefficient",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    author="Xialei Huang",
    description="Improved Chatterjee correlation coefficient and power comparison scripts.",
    url="https://github.com/petterHXL/Improved_Chatterjee_correlation_coefficient",
    python_requires=">=3.6",
) 