import math
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

# Define the filename of the input text file
input_file = r"..\\Inputfiles\\News Articles\\sport\\510.txt"  # Replace with the path to your input text file

# Read the text from the input file
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

num_sentences = len(sentences)

fsentiment_score = [1] * num_sentences

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
i=0
# Calculate the sentiment score compared to whole document
for sentence, sentiment, sentiment_score in sentence_sentiments:
    if sentiment == whole_text_sentiment:
        fsentiment_score[i] = 0.5
    else:
        fsentiment_score[i] = 0
    i=i+1
# Display the score of every sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1} Score: {fsentiment_score[i]}")

# Combine sentences with their combined scores, lengths, and count of category keyphrases
sentence_info_combined = list(zip(sentences, fsentiment_score))

# Sort sentences by their combined scores in descending order
sorted_sentence_info_combined = sorted(sentence_info_combined, key=lambda x: x[1], reverse=True)

# Define the percentage of sentences to include in the summary and Calculate the number of sentences to include based on the percentage
summary_percentage = 0.5
num_top_sentences = int(num_sentences * summary_percentage)

# # Change this to the number of top sentences you want to select
# num_top_sentences = 4  

# Display the top sentences based on combined scores
top_sentences = sorted(sorted_sentence_info_combined[:num_top_sentences], key=lambda x: sentences.index(x[0]))

# Store the top sentences in a list
summary_sentences = [sentence for sentence, _ in top_sentences]

# Join the summary sentences into a single string
summary = ' '.join(summary_sentences)

# Print the summary
print("Summary:")
print(summary)