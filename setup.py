import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seq2seq",
    version="0.0.5",
    author="Yukio Fukuzawa",
    author_email="y.fukuzawa@massey.ac.nz",
    description="Real valued sequence to sequence autoencoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fzyukio/multidimensional-variable-length-seq2seq-autoencoder",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=['numpy==1.16.3', 'tensorflow==2.6.4'],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
