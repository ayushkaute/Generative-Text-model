# Generative-Text-model

*COMPANY: CODTECH IT SOLUTIONS

*NAME: AYUSH MACHHINDRA KAUTE

*INTERN ID: CT04DF1740

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEEKS

*MENTOR: NEELA SANTOSH

DESCRIPTION:

This Python script demonstrates a simple LSTM-based text generation model using PyTorch. It includes a custom tokenizer, vocabulary builder, and a basic LSTM neural network to process a short sequence of words and output predictions for the next words. LSTM networks are a special kind of Recurrent Neural Network (RNN) capable of learning long-term dependencies, making them useful for tasks involving sequences like text, speech, or time series.

📜 1. Tokenization and Vocabulary Creation

def simple_tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().split()
    
This function cleans the input text by:
Removing punctuation using regular expressions.
Lowercasing the text.
Splitting it into individual words (tokens) using whitespace.
Next, we define two example sentences:


texts = ["AI is changing education", "Teachers use technology to enhance learning"]
These sentences are tokenized, and a vocabulary is built using a set of unique words. The vocabulary is mapped to integer indices via:


vocab = {word: idx for idx, word in enumerate(set(tokens))}
This mapping allows the model to convert between human-readable words and machine-readable numbers.

🔢 2. Encoding Function

def encode(text):
    return torch.tensor([vocab.get(word, 0) for word in simple_tokenizer(text)], dtype=torch.long)
    
The encode() function turns a sentence into a list of integers using the vocabulary mapping. If a word is not in the vocabulary, it defaults to index 0.

🧠 3. LSTM Model Definition

The core of the model is defined in the LSTMGenerator class:
class LSTMGenerator(nn.Module):

Embedding Layer: Converts input word indices into dense vectors of a specified dimension (embed_dim=128). Embeddings allow the model to learn semantic relationships between words.
self.embedding = nn.Embedding(vocab_size, embed_dim)

LSTM Layer: The heart of the model. It reads sequences of word embeddings and maintains memory of previous words, allowing it to predict contextually appropriate outputs.
self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

Fully Connected Layer: Projects the output from the LSTM’s hidden states to a vocabulary-sized output. Each output vector represents scores for possible next words in the vocabulary.
self.fc = nn.Linear(hidden_dim, vocab_size)
The forward() function performs the forward pass: it embeds the input, runs it through the LSTM, and transforms it through the linear layer to get predictions.

⚙️ 4. Running the Model
In the __main__ block, the model is initialized and used on an example input:

input_text = "AI is"
encoded = encode(input_text).unsqueeze(0)

The input string "AI is" is encoded and reshaped to match the model’s batch size expectations.
The model is run in evaluation mode (model.eval()), and it outputs a tensor representing predictions for each word in the sequence.

📊 Output
The final print statement displays the output tensor from the model:

print(output)
This output contains scores (logits) for each word in the vocabulary at each position in the input sequence. Though this output isn’t directly human-readable, it can be passed through a softmax function and used to select the most likely next word, enabling basic text generation.

✅ Summary:
This script provides a minimal, readable implementation of an LSTM-based model for text sequence processing. It shows:
How to preprocess text.
How to create and train simple NLP models.
The basics of LSTM networks and embedding layers.
This is a great starting point for building more advanced NLP systems like chatbots, autocomplete engines, or text generators.
