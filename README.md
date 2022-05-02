

# Extractive Text Summarization
## It contains the code to do extractive text summarization using the scores obtain by pagerank on the pairwise similarities between sentences. (Although there are many other approaches to follow)


## Running the script
```
python extractive_text_sm.py text.txt <num_sen>
```

where the text.txt contains sentences of the text document, all in one line and <num_sen> specifies number of sentences in the summary.

## Requirements for running the script:
1. Python 3
2. Python packages: numpy, re, sys, nltk, networkx
3. text file with all sentences on one line

## The requirements can be installed with
```
pip3 install -r requirements.txt
```
where the requirements.txt contains line seperated names of packages specified in point 2.
