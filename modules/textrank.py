import math
import nltk
from nltk.tokenize import sent_tokenize

# Define the filename of the input text file
input_file = r"..\\Inputfiles\\News Articles\\sport\\510.txt"  # Replace with the path to your input text file

# Read the text from the input file
with open(input_file, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

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
num_sentences = len(sentences)
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

# Display the score of every sentence
for i, sentence in enumerate(sentences):
    print(f"Sentence {i + 1} Score: {sentence_scores[i]}")

# Combine sentences with their combined scores, lengths, and count of category keyphrases
sentence_info_combined = list(zip(sentences, sentence_scores))

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