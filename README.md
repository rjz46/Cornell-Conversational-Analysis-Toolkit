# Cornell Conversational Analysis Toolkit ([ConvoKit](http://convokit.cornell.edu/))
This toolkit contains tools to extract conversational features and analyze social phenomena in conversations, using a [single unified interface](https://zissou.infosci.cornell.edu/convokit/documentation/architecture.html) inspired by (and compatible with) scikit-learn.  Several large [conversational datasets](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit#datasets) are included together with scripts exemplifying the use of the toolkit on these datasets. The latest version is [2.0.10](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/releases/tag/v2.0) (released 21 Aug 2019).

The toolkit currently implements features for:

### [Linguistic coordination](https://www.cs.cornell.edu/~cristian/Echoes_of_power.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/coordination.html)</sup></sub>

A measure of linguistic influence (and relative power) between individuals or groups based on their use of function words.  
Example: [exploring the balance of power in the U.S. Supreme Court](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/coordination/examples.ipynb).

### [Politeness strategies](https://www.cs.cornell.edu/~cristian/Politeness.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/politenessStrategies.html)</sup></sub>

A set of lexical and parse-based features correlating with politeness and impoliteness.  
Example: [understanding the (mis)use of politeness strategies in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb).

### [Conversational prompts](http://www.cs.cornell.edu/~cristian/Asking_too_much.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/questionTypology.html)</sup></sub>

An unsupervised method for extracting surface motifs that occur in conversations and grouping them by rhetorical role.  
Examples: [extracting common question types in U.K. parliament](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/question-typology/parliament_questions_example.ipynb), [understanding the use of conversational prompts in conversations gone awry on Wikipedia](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb).

### [Hypergraph conversation representation](http://www.cs.cornell.edu/~cristian/Patterns_of_participant_interactions.html) <sub><sup>[(API)](https://zissou.infosci.cornell.edu/convokit/documentation/hyperconvo.html)</sup></sub>
A method for extracting structural features of conversations through a hypergraph representation.  
Example: [hypergraph creation and feature extraction, visualization and interpretation on a subsample of Reddit](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/hyperconvo/demo.ipynb).

### [Linguistic diversity in conversations (Coming Soon!)](http://www.cs.cornell.edu/~cristian/Finding_your_voice__linguistic_development.html)
A method to compute the linguistic diversity of individuals within their own conversations, and between other individuals in a population.  

## Datasets
ConvoKit ships with several datasets ready for use "out-of-the-box".
These datasets can be downloaded using the `convokit.download()` [helper function](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/convokit/util.py).  Alternatively you can access them directly [here](http://zissou.infosci.cornell.edu/convokit/datasets/).

### [Conversations Gone Awry Dataset](https://zissou.infosci.cornell.edu/convokit/documentation/awry.html)

A collection of conversations from Wikipedia talk pages that derail into personal attacks (1,270 conversations, 6,963 comments)  
Name for download: `conversations-gone-awry-corpus`

### [Cornell Movie-Dialogs Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/movie.html)

A large metadata-rich collection of fictional conversations extracted from raw movie scripts. (220,579 conversational exchanges between 10,292 pairs of movie characters in 617 movies). 
Name for download: `movie-corpus`

### [Parliament Question Time Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/parliament.html)

Parliamentary question periods from May 1979 to December 2016 (216,894 question-answer pairs).  
Name for download: `parliament-corpus`

### [Supreme Court Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/supreme.html)

A collection of conversations from the U.S. Supreme Court Oral Arguments.  
Name for download: `supreme-corpus`

### [Wikipedia Talk Pages Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/wiki.html)

A medium-size collection of conversations from Wikipedia editors' talk pages.  
Name for download: `wiki-corpus`

### [Tennis Interviews](https://zissou.infosci.cornell.edu/convokit/documentation/tennis.html)

Transcripts for tennis singles post-match press conferences for major tournaments between 2007 to 2015 (6,467 post-match press conferences).  
Name for download: `tennis-corpus`


### [Reddit Corpus](https://zissou.infosci.cornell.edu/convokit/documentation/subreddit.html)

Reddit conversations from over 900k subreddits, arranged by subreddit. A [small subset](https://zissou.infosci.cornell.edu/convokit/documentation/reddit-small.html) sampled from 100 highly active subreddits is also available. 
 
Name for download: `subreddit-<name_of_subreddit>` for the by subreddit data, `reddit-corpus-small` for the small subset. 

### Wikiconv Corpus (WIP)

The full corpus of Wikipedia talk page conversations, based on the reconstruction described in [this paper](http://www.cs.cornell.edu/~cristian/index_files/wikiconv-conversation-corpus.pdf).
Note that due to the large size of the data, it is split up by year.
We are currently working on implementing, as part of the corpus metadata, block data retrieved directly from the Wikipedia block log, for reproducing the [Trajectories of Blocked Community Members](http://www.cs.cornell.edu/~cristian/Recidivism_online_files/recidivism_online.pdf) paper.
In the meantime, raw block data can be downloaded [here](https://zissou.infosci.cornell.edu/convokit/datasets/wikiconv-corpus/blocks.json).

Name for download: `wikiconv-<year>` to download wikiconv data for the specified year.

### ...And your own corpus!

In addition to the provided datasets, you may also use ConvoKit with your own custom datasets by loading them into a `convokit.Corpus` object. [This example script](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/converting_movie_corpus.ipynb) shows how to construct a Corpus from custom data.

## Installation
This toolkit requires Python >= 3.6.

1. Download the toolkit: `pip3 install convokit`
2. Download Spacy's English model: `python3 -m spacy download en`
3. Download NLTK's 'punkt' model: `import nltk; nltk.download('punkt')` (in Python interpreter)

Alternatively, visit our [Github Page](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit) to install from source.

## Documentation
Documentation is hosted [here](http://zissou.infosci.cornell.edu/convokit/documentation/). If you are new to ConvoKit, great places to get started are the [Core Concepts tutorial](https://zissou.infosci.cornell.edu/convokit/documentation/architecture.html) for an overview of the ConvoKit "philosophy" and object model, and the [High-level tutorial](https://zissou.infosci.cornell.edu/convokit/documentation/tutorial.html) for an walkthrough of how to import ConvoKit into your project, load a Corpus, and use ConvoKit functions.

## Contributing

We welcome community contributions. To see how you can help out, check the [contribution guidelines](https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/CONTRIBUTING.md).

## Citing

If you use the code or datasets distributed with ConvoKit please acknowledge the work tied to the respective component (indicated in the documentation) in addition to:

Jonathan P. Chang, Caleb Chiam, Liye Fu, Andrew Wang, Justine Zhang, Cristian Danescu-Niculescu-Mizil. 2019. "ConvoKit: The Cornell Conversational Analysis Toolkit" Retrieved from http://convokit.cornell.edu

[ConvoKit](http://convokit.cornell.edu/)
