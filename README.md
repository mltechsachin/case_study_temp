

# Extractive Text Summarization
## It contains the code to do extractive text summarization using the scores obtain by pagerank on the pairwise similarities between sentences. (Although there are many other approaches to follow)

## Approach
The main idea  here is to obtain similarities all pairs of sentences and returning the sentences having maximum similarity scores. 
I used cosine similarity as the similarity matrix and Textrank/Pagerank algorithm to rank the sentences based on their importance.

Steps involved here are as follows

1. The first step would be to concatenate all the text contained in the articles
2. Then split the text into individual sentences
3. Find vector representation (word embeddings) for every sentence pair, although other complex method like TF-IDF could be used
4. Similarities between sentence vectors are then calculated and stored in a matrix
5. The similarity matrix is then converted into a graph, with sentences as vertices and similarity scores as edges, for sentence rank calculation
6. Given number of top-ranked sentences form the final summary

## Running the script
```
python extractive_text_sm.py text.txt <num_sen>
```

where the text.txt contains sentences of the text document, all in one line and <num_sen> specifies number of sentences in the summary (3 by default).

## Requirements for running the script:
1. Python 3
2. Python packages: numpy, re, sys, nltk, networkx
3. text file with all sentences on one line

## The requirements can be installed with
```
pip3 install -r requirements.txt
```
where the requirements.txt contains line seperated names of packages specified in point 2.
