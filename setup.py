from setuptools import setup, find_packages

setup(
    name="QuantTools",        # Name of your package
    version="0.1",                 # Version number
    packages=find_packages(),      # Automatically find subpackages
    install_requires=[],           # Add dependencies if needed
    author="Chidi & Zino",            # Your name
    description="Repository for Quantitative Trading analysis",
    long_description=open("README.md").read(),  # Optional: Use your README as a description
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
