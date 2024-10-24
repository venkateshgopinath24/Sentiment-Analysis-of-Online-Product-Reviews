
# Sentiment Analysis of Online Product Reviews

## Introduction
This project performs a comprehensive sentiment analysis on **Amazon product reviews** using a combination of natural language processing (NLP) techniques and deep learning models. The analysis is conducted using two prominent techniques for sentiment scoring:
1. **VADER (Valence Aware Dictionary and Sentiment Reasoner)** from the NLTK library.
2. **RoBERTa (Robustly optimized BERT)**, a pretrained transformer model from Hugging Face.
3. **Sequential Neural Network**, Model build from scratch.

The **Sequential Neural Network** was constructed using **Keras**, where two types of word embeddings—**Average Word Embedding** and **GloVe Embedding**—were compared in their ability to predict sentiment. The aim of this project is to evaluate the effectiveness of these models in predicting customer sentiment from Amazon reviews and to compare their performance.

## Dataset
The dataset used is the **Amazon Customer Reviews** dataset, which includes over **3.7 million reviews** across various product categories. It consists of the following key fields:
- **Text**: The review content.
- **Score**: Star rating (1-5), where 5 indicates a highly positive sentiment and 1 indicates a negative sentiment.

The dataset is divided into three parts:
- **Training Set**: 60% of the data.
- **Validation Set**: 20% of the data.
- **Test Set**: 20% of the data.

The dataset was split using **stratified random sampling** to ensure balanced classes across each subset.

## Data Preprocessing
To prepare the text for analysis, the following preprocessing steps were applied:
- **Tokenization**: Split each review into individual words.
- **Lowercasing**: Converted all words to lowercase for consistency.
- **Stopword Removal**: Removed common words like "the", "is", and "in" that do not contribute to sentiment.
- **HTML Tag Removal**: Removed unnecessary HTML tags such as `<p>` and `<href>`.
- **Punctuation & Special Characters Removal**: Stripped all punctuation and special characters except periods.
- **Lemmatization**: Applied `WordNetLemmatizer` to reduce words to their base form.

### Length Thresholding
Through exploratory data analysis (EDA), we observed that most reviews contained fewer than **200 words**. Hence, a threshold of 200 words was applied, and reviews longer than this were excluded from the analysis.

## Vector Embedding
We employed **word embeddings** to represent the review text as numerical vectors. Two types of embeddings were used:
1. **Average Word Embedding**: The average embedding for each word in the review.
2. **GloVe Embedding**: Pre-trained **GloVe embeddings** (`glove.6B.100d.txt`), where each word is represented as a 100-dimensional vector.

These embeddings were then used as input to the **Keras Embedding Layer**.

## Sentiment Analysis Models

### A. VADER (NLTK)
**VADER** is a rule-based sentiment analysis tool that calculates polarity scores (negative, neutral, and positive). VADER is particularly effective for short, informal texts like **product reviews**.

**Key Features**:
- **Bag-of-words** approach: Each word is assigned a score, and the total sentiment score is computed by summing individual word scores.
- **Output**: Negative (`neg`), neutral (`neu`), positive (`pos`), and compound scores, where the compound score represents the overall sentiment.

### B. RoBERTa (Hugging Face)
**RoBERTa** (Robustly optimized BERT approach) builds on the BERT model but modifies key hyperparameters, improving performance on text classification tasks.

**Key Features**:
- **Transformer-based architecture**: Utilizes self-attention mechanisms to capture word dependencies and context within sentences.
- **Pretraining**: RoBERTa is pretrained on large text corpora, making it highly effective for downstream NLP tasks like sentiment analysis.

### C. Sequential Neural Networks (Keras)
We implemented a **Sequential Neural Network** using **Keras** to predict the sentiment of reviews. Two types of embeddings were fed into the model:
1. **Word Embeddings**: Directly created from the dataset.
2. **GloVe Embeddings**: Pre-trained embeddings that enhance word-level understanding.

**Model Architecture**:
- **Embedding Layer**: Converts input words into dense vector representations.
- **LSTM Layer**: Long Short-Term Memory units capture sequence dependencies.
- **Dense Output Layer**: Produces the final classification (positive or negative sentiment).

The sequential model is used for both average word embedding and GloVe embedding data frames.

## Loss Function
Given the binary nature of the problem (positive vs negative sentiment), the **binary cross-entropy** loss function is used to measure the error between predicted and actual labels.

$$
L_{binary} = -\left( y \log(p) + (1 - y) \log(1 - p) \right)
$$


## Evaluation Metrics
We evaluate the models using the **accuracy** metric for the training, validation, and test datasets. The key comparison is between VADER, RoBERTa, and the sequential models.

### Results:
- **VADER**: VADER achieved a moderate accuracy of around 70%, showing limitations in understanding complex text relationships.
- **RoBERTa**: RoBERTa significantly outperformed VADER with an accuracy of ~85%, thanks to its superior ability to capture contextual meaning.
- **Neural Networks**:
  - **Average Word Embeddings**: Achieved ~80% accuracy but faced some overfitting issues.
  - **GloVe Embeddings**: Performed slightly better, with ~82% accuracy, and showed better generalization on unseen data.

## Results Visualization

### VADER vs RoBERTa Performance
- The **VADER** model produced polarity scores (positive, neutral, and negative) and showed moderate correlation with the review scores.
- **RoBERTa** demonstrated a more accurate representation of review sentiment, aligning closely with the star ratings provided by customers.

### Neural Networks
- The neural networks trained with **GloVe embeddings** showed a smoother loss convergence curve compared to the **word embeddings**, leading to better generalization.
- **Training and Validation Loss/Accuracy** plots demonstrated the learning performance of the models across epochs.

## Conclusion
The **RoBERTa model** outperformed both VADER and the neural network models, proving to be the most effective approach for sentiment analysis of product reviews. The **GloVe-embedded sequential model** also performed well, providing a viable option for scenarios requiring word-level understanding.

## Future Work
- Implementing **Recurrent Neural Networks (RNNs)** for better sequence modeling of text.
- Extending the analysis to more transformer-based models, such as **GPT**.
- Applying **multi-modal analysis** by combining text-based sentiment with other inputs, such as image data from product listings.

## References
1. [Keras Sequential Model API](https://keras.io/api/models/sequential/)
2. [NLTK VADER Sentiment Analysis](https://www.nltk.org/api/nltk.sentiment.vader.html)
3. [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

