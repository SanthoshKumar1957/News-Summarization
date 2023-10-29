import nltk
import math
# nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from rake_nltk import Rake
import re  # Import the regular expressions library

# Define the filename of the input text file
input_file = r"..\\Inputfiles\\News Articles\\sport\\510.txt"  # Replace with the path to your input text file

# Read the text from the input file
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text into sentences and words
sentences = sent_tokenize(text)

num_sentences = len(sentences)

keyphrase_score = [1] * num_sentences

# Initialize a Rake object to extract keyphrases
r = Rake()

# Extract keywords (keyphrases) from the text using Rake
r.extract_keywords_from_text(text)

# Get the ranked keyphrases without duplicates
keyphrases = list(set(r.get_ranked_phrases()))

print(keyphrases)

# Initialize a Counter to count category keyphrases in sentences
sentence_category_keyphrase_count = Counter()

# Calculate the length of each sentence (counting only words)
sentence_lengths = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences]

i=0
# Calculate the score of each sentence using the formula
sentence_scores = []
for sentence in sentences:
    words_in_sentence = word_tokenize(sentence)
    count_category_keyphrases = sum(1 for word in words_in_sentence if word.lower() in map(str.lower, keyphrases))
    sentence_category_keyphrase_count[sentence] = count_category_keyphrases
    keyphrase_score[i] =  sentence_category_keyphrase_count[sentence] / sentence_lengths[i]
    sentence_scores.append(keyphrase_score[i])
    i=i+1

# # Display sentences with their scores, lengths, and count of category keyphrases
# for sentence, score, length in sorted_sentence_info:
#     print(f"Sentence: {sentence}")
#     print(f"Score: {score}")
#     print(f"Length: {length} words")
#     print(f"Count of Category Keyphrases: {sentence_category_keyphrase_count[sentence]}")
#     print()

# Display the score of every sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1} Score: {keyphrase_score[i]}")

# Combine sentences with their combined scores, lengths, and count of category keyphrases
sentence_info_combined = list(zip(sentences, keyphrase_score, sentence_lengths))

# Sort sentences by their combined scores in descending order
sorted_sentence_info_combined = sorted(sentence_info_combined, key=lambda x: x[1], reverse=True)

# Define the percentage of sentences to include in the summary and Calculate the number of sentences to include based on the percentage
summary_percentage = 0.4
num_top_sentences = int(num_sentences * summary_percentage)

# # Change this to the number of top sentences you want to select
# num_top_sentences = 4  

# Display the top sentences based on combined scores
top_sentences = sorted(sorted_sentence_info_combined[:num_top_sentences], key=lambda x: sentences.index(x[0]))

# Store the top sentences in a list
summary_sentences = [sentence for sentence, _, _ in top_sentences]

# Join the summary sentences into a single string
summary = ' '.join(summary_sentences)

# Print the summary
print("Summary:")
print(summary)