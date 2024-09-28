import re
from concurrent.futures import ThreadPoolExecutor
from langdetect import detect
import numpy as np

# Cache for repeated tokenization results
token_cache = {}

# Example Vocabulary, Morphology Rules, and Phrase Dictionary
vocabulary = {"the", "cat", "sat", "on", "mat", "##ting", "##ted", "run", "walk"}
morphology_rules = {
    "running": ["run", "##ning"],
    "cats": ["cat", "##s"],
}
phrase_dict = {
    "on the": "[on_the]",
    "the mat": "[the_mat]",
}

word_frequencies = {
    "the": 10000, "cat": 500, "sat": 400, "on": 800, "mat": 100, "running": 50
}

# Level 1: Basic Whitespace Tokenizer (Word-Level Tokenization)
def basic_word_tokenizer(text):
    return text.split()

# Level 2: Subword Tokenizer (Byte-Pair Encoding - BPE or WordPiece-style)
def subword_tokenizer(tokens, vocabulary):
    refined_tokens = []
    for token in tokens:
        if token in vocabulary:
            refined_tokens.append(token)
        else:
            refined_tokens.extend(split_into_subwords(token, vocabulary))
    return refined_tokens

def split_into_subwords(token, vocabulary):
    subwords = []
    i = 0
    while i < len(token):
        found_subword = None
        for j in range(i+1, len(token)+1):
            subword_candidate = token[i:j]
            if subword_candidate in vocabulary:
                found_subword = subword_candidate
        if found_subword:
            subwords.append(found_subword)
            i += len(found_subword)
        else:
            subwords.append(token[i])
            i += 1
    return subwords

# Level 3: Morphology-Aware Tokenizer
def morphology_aware_tokenizer(tokens, morphology_rules):
    refined_tokens = []
    for token in tokens:
        if token in morphology_rules:
            refined_tokens.extend(morphology_rules[token])
        else:
            refined_tokens.append(token)
    return refined_tokens

# Level 4: Semantic Tokenizer (Grouping idioms, phrases, etc.)
def semantic_tokenizer(tokens, phrase_dict):
    refined_tokens = []
    skip_next = False
    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        two_word_phrase = ' '.join(tokens[i:i+2])
        if two_word_phrase in phrase_dict:
            refined_tokens.append(phrase_dict[two_word_phrase])
            skip_next = True
        else:
            refined_tokens.append(tokens[i])
    return refined_tokens

# Dynamic Task-Specific Tokenization Adjustment
def dynamic_tokenization_adjustment(task, tokens):
    if task == "summarization":
        print("Skipping to semantic tokenization for summarization task...")
        tokens = semantic_tokenizer(tokens, phrase_dict)
    elif task == "question_answering":
        print("Focusing on subword tokenization for question answering task...")
        tokens = subword_tokenizer(tokens, vocabulary)
    return tokens

# Language Detection and Task-Specific Processing
def language_aware_tokenizer(text):
    language = detect(text)
    tokens = basic_word_tokenizer(text)
    if language == 'en':
        tokens = subword_tokenizer(tokens, vocabulary)
    elif language == 'tr':  # Turkish example
        tokens = morphology_aware_tokenizer(tokens, morphology_rules)
    else:
        tokens = dynamic_granularity_tokenizer(tokens, language)
    return tokens

# Dynamic Granularity Control (Applies subword tokenization based on token length or task)
def dynamic_granularity_tokenizer(tokens, task):
    refined_tokens = []
    for token in tokens:
        if len(token) > 6 and task != "summarization":
            refined_tokens.extend(subword_tokenizer([token], vocabulary))
        else:
            refined_tokens.append(token)
    return refined_tokens

# Embedding-Aware Tokenization (Uses embeddings to decide tokenization level)
def embedding_aware_tokenizer(tokens, embedding_model):
    refined_tokens = []
    for token in tokens:
        if token in embedding_model:
            refined_tokens.append(token)
        else:
            refined_tokens.extend(subword_tokenizer([token], vocabulary))
    return refined_tokens

# Attention-based Tokenization (Dynamic tokenization based on attention scores)
def attention_based_tokenizer(tokens, attention_scores):
    refined_tokens = []
    for token, score in zip(tokens, attention_scores):
        if score > 0.5:
            refined_tokens.extend(subword_tokenizer([token], vocabulary))
        else:
            refined_tokens.append(token)
    return refined_tokens

# Frequency-Based Subword Refinement
def frequency_based_tokenizer(tokens):
    refined_tokens = []
    for token in tokens:
        if word_frequencies.get(token, 0) < 100:  # Apply subword for rare words
            refined_tokens.extend(subword_tokenizer([token], vocabulary))
        else:
            refined_tokens.append(token)
    return refined_tokens

# Parallelized Tokenization for large inputs
def parallel_tokenization(text):
    chunks = [text[i:i+50] for i in range(0, len(text), 50)]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(basic_word_tokenizer, chunks))
    return [token for result in results for token in result]  # Flatten the results

# Cache-Based Tokenization
def cache_based_tokenizer(text):
    if text in token_cache:
        return token_cache[text]
    tokens = basic_word_tokenizer(text)
    token_cache[text] = tokens  # Store in cache
    return tokens

# Hierarchical Task-Based Pre-Processing
def hierarchical_preprocessing(text, task):
    tokens = basic_word_tokenizer(text)
    if task == "summarization":
        tokens = semantic_tokenizer(tokens, phrase_dict)
    elif task == "question_answering":
        tokens = subword_tokenizer(tokens, vocabulary)
    elif task == "translation":
        tokens = morphology_aware_tokenizer(tokens, morphology_rules)
    else:
        tokens = dynamic_granularity_tokenizer(tokens, task)
    return tokens

# Final Hierarchical Tokenization with Dynamic Features
def hierarchical_tokenization(text, task=None):
    print(f"Original Text: {text}")

    # Check for cached result
    tokens = cache_based_tokenizer(text)

    # Branching into dynamic tokenization based on task
    if task:
        tokens = dynamic_tokenization_adjustment(task, tokens)
    else:
        # Level 2: Subword Tokenization
        tokens = subword_tokenizer(tokens, vocabulary)
        print("Level 2 Tokens (Subwords):", tokens)

        # Level 3: Morphology-Aware Tokenization
        tokens = morphology_aware_tokenizer(tokens, morphology_rules)
        print("Level 3 Tokens (Morphology):", tokens)

        # Level 4: Semantic Tokenization
        tokens = semantic_tokenizer(tokens, phrase_dict)
        print("Level 4 Tokens (Semantic):", tokens)

    return tokens

# Example Usage
text_input = "The cat sitting on the mat was running."

# Running for a general context without task-specific branching
print("\nGeneral Context Tokenization:")
tokens_general = hierarchical_tokenization(text_input.lower())

# Running for a summarization task, which skips to semantic tokenization
print("\nSummarization Task Tokenization:")
tokens_summary = hierarchical_tokenization(text_input.lower(), task="summarization")

# Running for a question answering task, which focuses on subword tokenization
print("\nQuestion Answering Task Tokenization:")
tokens_qa = hierarchical_tokenization(text_input.lower(), task="question_answering")
