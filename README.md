Multipass_LLM
Python implementation of a 4-level multi-pass tokenization with branching into revised tokenization strategies

Key Features of the Code:

1. Level 1 (Basic Word-Level Tokenization):
    • This level splits the text into basic word tokens using spaces. It’s a fast, simple pass to break down sentences into individual components.
2. Level 2 (Subword Tokenization):
    • This pass uses a vocabulary of known words and subword units, handling rare or unseen words by breaking them into subword tokens (using a simplified Byte-Pair Encoding approach). Words that are not in the vocabulary are split into known subwords or single characters.
3. Level 3 (Morphology-Aware Tokenization):
    • This level breaks words into meaningful morphological components. For example, “running” becomes [“run”, “##ning”]. This is useful for morphologically rich languages or complex word forms.
4. Level 4 (Semantic Tokenization):
    • This stage groups multi-word expressions, idiomatic phrases, or common bigrams into single tokens, allowing the model to understand meaning at a higher semantic level (e.g., “on the” → “[on_the]”). This helps in preserving the contextual meaning of phrases.
5. Dynamic Tokenization Adjustments:
    • Depending on the task (e.g., summarization or question answering), the tokenization strategy can change:
    • Summarization: Skips directly to semantic tokenization, as summarization benefits from understanding overall meaning and phrases.
    • Question Answering: Focuses on subword tokenization to handle potential rare or complex words that are likely to appear in queries.
6. Branching and Hierarchical Processing:
    • Each tokenization pass builds upon the previous one, but if a specific task is provided, the model branches dynamically into the appropriate tokenization pass to optimize the output.

Key Updates and Features Added:

1. Dynamic Granularity Control: The tokenizer adjusts the granularity of tokenization (word-level or subword) based on the input length and the task.
2. Embedding-Aware Tokenization: Uses pre-trained embeddings to refine tokenization. Tokens not present in the embedding space are split into subwords.
3. Attention-Based Tokenization: Incorporates attention scores to apply more granular tokenization to high-attention tokens.
4. Language-Aware Tokenization: Detects the language of the input and applies appropriate tokenization strategies, such as morphology-aware tokenization for morphologically rich languages.
5. Parallelized Tokenization: Processes large input sequences in parallel, improving the speed for long documents.
6. Cache-Based Tokenization: Uses a cache to store previously tokenized texts, improving efficiency when processing repeated inputs.
7. Frequency-Based Subword Refinement: Applies subword tokenization to less frequent words, reducing the token count for common words and ensuring proper handling of rare terms.
