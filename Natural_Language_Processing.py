# First, we will explore the CSV file to determine what type of data we can use for the analysis and how it is structured. 
# A research paper typically consists of a title, an abstract and the main text. 
# Other data such as figures and tables were not extracted from the PDF files because it does not have trend discussions. 

import pandas as pd

papers = pd.read_csv("datasets/papers.csv")
papers.head(5)

#We will be processing the data to filter what we need, which is the year of publication and the text data.
#Metadata such as id's and filenames will be removed because it has no influence on the input variable
#We remove the metadata columns using the .drop function
papers.drop(["id","event_type","pdf_name"],axis=True, inplace=True)


#As a primary indication of the first trend, we must account for the number of papers published by year
# Group the papers by year using the .groupby function
groups = papers.groupby(['year'])

# Determining the size of each group to enable data visualiation
counts = groups.size()

# Data viz
import matplotlib.pyplot 
get_ipython().run_line_magic('matplotlib', 'inline')

counts.plot(kind='bar')



#Here we are analyzing the titles for AI research trends
#We remove any punctuation and capitalization from the titles so that the theme can be captured more accurately

#We will use the regular expressions library
import re 
print(papers['title'].head(3))

# Remove punctuation
papers['title_processed'] = papers['title'].map(lambda x: re.sub('[,\.!?]', '', x))

# Convert the titles to lowercase
papers['title_processed'] = papers['title'].map(lambda x:x.lower())
papers['title_processed'].head()                                                                


#In order to verify whether the preprocessing happened correctly, we will make a word cloud of the titles of the research papers. 
# This will give us a visual representation of the most common words.  
# In addition, it allows us to verify whether we need additional preprocessing before further analyzing the data.

import wordcloud
# Join the different processed titles together by removing spaces
long_string = ''.join(papers['title_processed'])

# Creating a WordCloud object
wordcloud = wordcloud.WordCloud()

# Generate a word cloud
wordcloud.generate(long_string)
# Visualizing the word cloud
wordcloud.to_image()


#LDA does not work directly on text data. 
# First, it is necessary to convert the documents into a simple vector representation. 
# This representation will then be used by LDA to determine the topics. 
# Each entry of a 'document vector' will correspond with the number of times a word occurred in the document. 
# In conclusion, we will convert a list of titles into a list of vectors, all with length equal to the vocabulary. 
# We'll then plot the 10 most common words based on the outcome of this operation (the list of document vectors). 
# These words should also occur in the word cloud.</p>

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 

    plt.bar(x_pos, counts,align='center')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.title('10 most common words')
    plt.show()

# Initialise the count vectorizer with the English stop words
count_vectorizer =CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(papers['title_processed'])

# Visualise the 10 most common words
plot_10_most_common_words(count_data, count_vectorizer)


#Analysing trends with LDA
# LDA is able to perform topic detection on large document sets, determining what the main 'topics' are in a large unlabeled set of texts. 
#LDA will note the trends which it associates with consistent topics
#A topic is termed as a set of words that seem to occur commonly together
#Note, however, that when applying new AI papers, the data must be processed like the training data above.
#The output will be what the topics are about

#Here we will suppress the warnings that sklearn may generate
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from sklearn.decomposition import LatentDirichletAllocation as LDA

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below using the perplexity metric (use integer values below 15) 
number_topics = 10
number_words = 10

# Create and fit the LDA model
lda = LDA(n_components=number_topics)
lda.fit(count_data)

# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

#Further analysis can be done on this data to expose more interesting details