from setuptools import setup, find_packages

setup(
    name = "convokit",
    author = "Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil",
    author_email = "cristian@cs.cornell.edu",
    url = "https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit",
    description = "Cornell Conversational Analysis Toolkit",
    version = "2.0.10",
    packages = ["convokit",
                "convokit.coordination",
                "convokit.hyperconvo",
                "convokit.model",
                "convokit.parser",
                "convokit.politeness_api",
                "convokit.politeness_api.features",
                "convokit.politenessStrategies",
                "convokit.questionTypology"],
    package_data = {"convokit": ["data/*.txt"]},
    install_requires = [
        "matplotlib>=3.0.0",
        "pandas>=0.23.4",
        "msgpack-numpy==0.4.3.2",
        "spacy==2.0.12",
        "scipy>=1.1.0",
        "scikit-learn>=0.20.0",
        "nltk>=3.4",
        "dill==0.2.9"
    ]
)
