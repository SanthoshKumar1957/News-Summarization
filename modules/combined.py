import math
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re
import requests  # Import the requests library
from bs4 import BeautifulSoup  # Import BeautifulSoup
# nltk.download('stopwords')  # Uncomment this line if you haven't downloaded stopwords

# #Paste the text file
# text=""

# Define the filename of the input text file
input_file = r"..\\Inputfiles\\News Articles\\sport\\510.txt"  # Replace with the path to your input text file

# Read the text from the input file
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

# from newspaper import Article

# # URL of the news article
# url = 'https://www.ndtv.com/world-news/israel-gaza-palestine-as-war-escalates-arab-countries-say-palestinians-must-stay-on-their-land-4480701#News_Trending'

# # Initialize the Article object
# article = Article(url)

# # Download and parse the article
# article.download()
# article.parse()

# # Get the main article text
# text = article.text

# #Print the text
# print(text)

# Tokenize the text into sentences
sentences = sent_tokenize(text)

num_sentences = len(sentences)

# Function to calculate cosine similarity between two sentences
def cosine_similarity(sentence1, sentence2):
    words1 = set(sentence1.lower().split())
    words2 = set(sentence2.lower().split())
    common_words = words1.intersection(words2)
    
    # Calculate the dot product of the sentence vectors
    dot_product = sum(1 for word in common_words)
    
    # Calculate the magnitude of each sentence vector
    magnitude1 = math.sqrt(len(words1))
    magnitude2 = math.sqrt(len(words2))
    
    # Calculate cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)
    
    return similarity

# Create a similarity matrix
similarity_matrix = [[0] * num_sentences for _ in range(num_sentences)]

for i in range(num_sentences):
    for j in range(i, num_sentences):
        similarity = cosine_similarity(sentences[i], sentences[j])
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

# Implement the Textrank algorithm to calculate sentence scores
damping_factor = 0.85
max_iterations = 100
sentence_scores = [1] * num_sentences

for iteration in range(max_iterations):
    new_scores = []
    for i in range(num_sentences):
        new_score = (1 - damping_factor)
        for j in range(num_sentences):
            if i != j:
                new_score += damping_factor * (similarity_matrix[i][j] / sum(similarity_matrix[j]))
        new_scores.append(new_score)
    sentence_scores = new_scores

# Create a TextBlob object for sentiment analysis
blob = TextBlob(text)

# Calculate sentiment for every sentence
sentence_sentiments = []

for sentence in blob.sentences:
    sentiment_score = sentence.sentiment.polarity
    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    sentence_sentiments.append((str(sentence), sentiment, sentiment_score))

# Calculate sentiment for the whole text
whole_text_sentiment_score = blob.sentiment.polarity

if whole_text_sentiment_score > 0:
    whole_text_sentiment = "Positive"
elif whole_text_sentiment_score < 0:
    whole_text_sentiment = "Negative"
else:
    whole_text_sentiment = "Neutral"

# Calculate the sentiment score compared to the whole document
fsentiment_score = [1] * num_sentences

i = 0
for sentence, sentiment, sentiment_score in sentence_sentiments:
    if sentiment == whole_text_sentiment:
        fsentiment_score[i] = 0.5
    else:
        fsentiment_score[i] = 0
    i += 1

# Initialize a Rake object to extract keyphrases
r = Rake()

# Extract keywords (keyphrases) from the text using Rake
r.extract_keywords_from_text(text)

# Get the ranked keyphrases without duplicates
keyphrases = list(set(r.get_ranked_phrases()))

# Initialize a Counter to count category keyphrases in sentences
sentence_category_keyphrase_count = Counter()

# Calculate the length of each sentence (counting only words)
sentence_lengths = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]

keyphrase_score = [1] * num_sentences

i = 0
# Calculate the score of each sentence using the formula
for sentence in sentences:
    words_in_sentence = word_tokenize(sentence)
    count_category_keyphrases = sum(1 for word in words_in_sentence if word.lower() in map(str.lower, keyphrases))
    sentence_category_keyphrase_count[sentence] = count_category_keyphrases
    keyphrase_score[i] =  sentence_category_keyphrase_count[sentence] / len(words_in_sentence)
    i += 1

# Define the weights for each score
weight_sentence = 1  # Weight for sentence score
weight_keyphrase = 1 # Weight for keyphrase score
weight_sentiment = 1/3  # Weight for sentiment score

# Initialize a list to store the combined scores
combined_scores = []

# Calculate the combined score for each sentence
for i in range(num_sentences):
    combined_score = (
        weight_sentence * sentence_scores[i] +
        weight_keyphrase * keyphrase_score[i] +
        weight_sentiment * fsentiment_score[i]
    )

    combined_scores.append(combined_score)

i=0
# Display the score of every sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1} Score: {combined_scores[i]}")
    i=i+1

# Combine sentences with their combined scores, lengths, and count of category keyphrases
sentence_info_combined = list(zip(sentences, combined_scores, sentence_lengths))

# Sort sentences by their combined scores in descending order
sorted_sentence_info_combined = sorted(sentence_info_combined, key=lambda x: x[1], reverse=True)

# Define the percentage of sentences to include in the summary and Calculate the number of sentences to include based on the percentage
summary_percentage = 0.5
num_top_sentences = int(num_sentences * summary_percentage)

# # Change this to the number of top sentences you want to select
# num_top_sentences = 4  

# Display the top sentences based on combined scores
top_sentences = sorted(sorted_sentence_info_combined[:num_top_sentences], key=lambda x: sentences.index(x[0]))

# # Display the top sentences in their original order
# for i, (sentence, score, length) in enumerate(top_sentences):
#     print(f"Top Sentence {i + 1} - Score: {score}")
#     print(f"Sentence: {sentence}")
#     # print(f"Length: {length} words")
#     print()

# Store the top sentences in a list
summary_sentences = [sentence for sentence, _, _ in top_sentences]

# Join the summary sentences into a single string
summary = ''.join(summary_sentences)

# Print the summary
print("Summary:")
print(summary)