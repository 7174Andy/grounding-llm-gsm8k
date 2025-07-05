import nltk.translate.bleu_score as bleu

def calculate_BLEU(generated_text, reference_text):
    # Tokenize the sentences
    generated_tokens = generated_text.split()
    reference_tokens = reference_text.split()

    # Calculate BLEU score
    score = bleu.sentence_bleu([reference_tokens], generated_tokens)
    return score
