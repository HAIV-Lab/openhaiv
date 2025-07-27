from setuptools import setup, find_packages

setup(
    name="ncdia",
    version="0.1",
    description="ncdia",
    packages=find_packages(),
    python_requires=">=3.8",
    package_data={
        "ncdia.models.clip_based.clip": ["bpe_simple_vocab_16e6.txt.gz"],
    },
)
