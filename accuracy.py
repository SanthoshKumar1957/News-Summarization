from rouge import Rouge

def calculate_metrics(generated_summary, reference_summary):
    # Preprocessing: Tokenization, lowercasing, and removing punctuation
    generated_summary = generated_summary.lower().split()
    reference_summary = reference_summary.lower().split()

    # Calculate precision
    common_words = set(generated_summary) & set(reference_summary)
    precision = len(common_words) / len(generated_summary)

    # Calculate recall
    recall = len(common_words) / len(reference_summary)

    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

# Calculate ROUGE scores
def calculate_rouge(generated_summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

# Example usage
generated_summary = "Claxton hunting first major medal.British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid. Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage. For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form. In previous seasons, the 25-year-old also contested the long jump but since moving from Colchester to London she has re-focused her attentions."
reference_summary = "For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form.Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage.British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March.\"I am quite confident,\" said Claxton.."

precision, recall, f1_score = calculate_metrics(generated_summary, reference_summary)
rouge_scores = calculate_rouge(generated_summary, reference_summary)

print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
print("ROUGE Scores:", rouge_scores)
