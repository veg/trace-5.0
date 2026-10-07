from setuptools import setup, find_packages

setup(
    name="trace50",
    version="5.0.0",
    description="Dynamic Molecular Surveillance of HIV-1 Transmission Chains at Registry Scale",
    author="Steven Weaver, Darren P. Martin, Joel O. Wertheim, Sergei L. Kosakovsky Pond",
    author_email="spond@temple.edu",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0"
    ],
    package_data={
        "trace50": ["data/reference_panels/*"]
    },
    entry_points={
        "console_scripts": [
            "trace50 = trace50.cli:main",
        ],
    },
    python_requires=">=3.8",
)
