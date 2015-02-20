from collections import Counter
from itertools import combinations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import TruncatedSVD

def load_data(filename,cutoff=5,nodigits=True):
    "Load list of tagsets with tags occurring more than the cutoff value."
    
    # open the file and get a list of tagsets:
    with open(filename) as f:
        tags    = [line.split()[1:] for line in f.readlines()]
    
    # create a counter to determine the common tags:
    c           = Counter([t for l in tags for t in l])
    
    # remove digits:
    if nodigits:
        digits  = [tag for tag in c if tag.isdigit()]
        for digit in digits:
            c.pop(digit)
    
    # get a list of the common tags:
    common_tags = set([tag for tag in c if c[tag] > cutoff])
    
    # define an internal function to select common tags:
    def common_string(t):
        "Remove infrequent tags from the tag string."
        return ' '.join(list(set(t) & common_tags))
    
    # return a list of tagsets, formatted as tag-separated strings:
    return [common_string(t) for t in tags]

def termterm(documents):
    """Function to create a toy model.
    Creates a term*term matrix, and performs TFIDF.

    Input: a list of documents. Each document should be a space-separated string.
    Output: a matrix and a list of row labels."""
    # First create CountVectorizer to build a Document-Term matrix.
    vec      = CountVectorizer(tokenizer=lambda s:s.split())
    data     = vec.fit_transform(documents)   # create sparse representation
    word_list = vec.get_feature_names()   # get list of tags
    termterm = data.transpose() * data   # compute term*term matrix
    vect     = TfidfTransformer()        # create TfidfTransformer
    tfidf    = vect.fit_transform(termterm) # ..and fit it to the term-term matrix
    return (tfidf,word_list)

def reduce_matrix(matrix,dim=400):
    "Perform SVD with the given number of dimensions. Returns reduced matrix."
    svd     = TruncatedSVD(n_components=dim)
    return svd.fit_transform(matrix)

def get_toy_data():
    """Function returning weighted edges to play with."""
    tags             = load_data('data/sfx_tags.txt')
    matrix,word_list = termterm(tags)
    reduced          = reduce_matrix(matrix,400)
    d                = pairwise_distances(reduced,metric='cosine')
    gen              = combinations(range(len(word_list)),2)
    for x,y in gen:
        yield (word_list[x],word_list[y],d[x][y])
