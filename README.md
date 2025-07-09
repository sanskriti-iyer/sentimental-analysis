# sentimental-analysis
__Overview__
This project applies natural language processing (NLP) and machine learning techniques to classify the sentiment (positive, neutral, negative) of airline-related tweets. Although the title mentions IMDB reviews, the actual dataset used is a subset of 4,000 tweets labeled for sentiment. The pipeline includes preprocessing, vectorization, model training, and evaluation.

__Core Objectives__
1. Classify tweets into three sentiment classes: positive, neutral, and negative
2. Build and evaluate multiple ML models using TF-IDF and CountVectorizer
3. Apply tokenization, stopword removal, POS tagging, and lemmatization for text cleaning

# Workflow Summary
__Data Loading__
- Imported tweets dataset (Tweets.csv) from Google Drive using pandas
- Subset created using the first 4,000 records for modeling

__Text Preprocessing__
- Downloaded essential NLTK resources (tokenizers, stopwords, lemmatizer)
- Tokenized tweets using word_tokenize
- Removed punctuation and stopwords
- Tagged tokens with parts-of-speech (POS)
- Performed lemmatization using WordNet lemmatizer based on POS

__Feature Engineering__
- Converted cleaned tokens back into sentences
- Generated feature vectors using both CountVectorizer and TfidfVectorizer

__Model Training__
- Trained and evaluated multiple classifiers:
- Support Vector Classifier (SVC)
- Multinomial Naive Bayes
- Random Forest Classifier
- Split data using train_test_split, trained on the training portion, and predicted on test data.

__Evaluation__
- Compared model performance using accuracy and classification reports
- Tested generalization ability on unseen data

# Technology Used
- Pandas – for loading and manipulating tweet data
- NumPy – for numerical operations and array handling
- NLTK (Natural Language Toolkit) – for tokenization, stopword removal, POS tagging, and lemmatization
- Scikit-learn (sklearn) – for vectorization (TF-IDF, CountVectorizer), model training (SVC, Naive Bayes, Random Forest), and evaluation
- Google Colab – as the development and execution environment
