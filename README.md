# Small Scale LLMs

The goal is to create small-scale language models from scratch using custom datasets, and also leverage Hugging Face's GPT-Neo model through transfer learning to fine-tune on specific data.

## Knowledge

```
Let’s say the model is given the sentence: "The cat sat on the mat."
```

1. Tokenization: 
    The sentence is tokenized into: ["The", "cat", "sat", "on", "the", "mat", "."]

2. Embedding:
    Each token is transformed into a vector representation (based on learned parameters). For example:
        "The" → [0.1, 0.3, -0.5, ...] (an embedding vector of numbers)
        "cat" → [0.2, -0.1, 0.4, ...]

3. Model Processing:
    The model's parameters are used to process these embeddings through layers like attention and feedforward networks. The parameters adjust how the model interprets the sentence, understanding that "cat" is an animal and "mat" is an object.

4. Output:
    The model might predict the next word based on the context or classify the sentence (e.g., is it positive or negative?). This prediction is guided by the weights (parameters) the model has learned during training.


**What is means by GPT Neo (125M) ?**

1. Tokens: These are the actual input units (words, subwords, or symbols) fed into the model.
2. Parameters: These are the learned weights inside the model that determine how the input tokens are processed and understood. They adjust the mathematical transformations applied to the tokens, but they are not the tokens themselves.

By 125M, we are pointing on 125 millions parameters


## FAQs From ChatGPT

1. How Many Prior Words (Context) Are Taken for Prediction?

Context Window: GPT-based models, including ChatGPT, rely on a context window, which is a fixed number of tokens (words, punctuation, etc.) that the model considers when predicting the next word. For example, GPT-3's context window is 4096 tokens, meaning it can look back at up to 4096 tokens to determine the next word.

2. It's not merely retrieving memorized responses but generating text based on statistical patterns.

When I say that a model like ChatGPT isn't just "retrieving memorized responses" but generating text based on statistical patterns, I mean that it doesn't simply repeat back sentences it has seen during training. Instead, it uses what it has learned about how language works to generate new sentences based on the input it receives.

```
Example
Let’s say the model was trained on a sentence like:

Sentence 1:

"The cat is sitting on the mat."

Now, if you give me a new input like:

Input:

"The dog is lying on the..."
```

I haven't specifically memorized a sentence like "The dog is lying on the floor." But based on the statistical patterns I've learned from millions of sentences, I understand that:

After "lying on the...", it’s likely that a noun like "ground," "sofa," or "floor" will follow.
Dogs often lie on soft surfaces like "carpets" or "beds."

3. How does GPT codes?

Common Functions and Libraries: I’ve learned the names and usage of common functions and libraries (like print() in Python or std::cout in C++). I don’t “understand” what these functions do at a deep level like a human programmer might, but I’ve seen them enough to recognize how they’re used.

4. Training and tokens size?

GPT-3: The predecessor to GPT-4 was trained on about 45 terabytes of text data before pre-processing.
The size of my vocabulary is in the hundreds of thousands of words.

5. Dealing with Unknown Terms

Proper Nouns: I handle most proper nouns (e.g., place names, personal names) well because they're often treated as tokens in the model, but if an entirely new or made-up word appears, I can still handle it through subword tokenization and context-based inference.

Misspelled Words: If a word is misspelled, my subword tokenizer helps break it into meaningful parts, and I can often understand it in context.

Symbols and Numbers: My vocabulary includes symbols, mathematical expressions, and numbers. If a symbol or unusual number is outside my immediate vocabulary, subword tokenization can help me process it appropriately.

```
# Sample word that may not be in vocabulary
input_text = "What is the quantum property of a zentiflexor?"
```
Even if "zentiflexor" is not in the vocabulary, I can break it down into subwords like "zenti" and "flexor," and process the meaning based on the surrounding context ("quantum property").

## Results:

![Gradio in progress](screenshots/screenshot.png)

