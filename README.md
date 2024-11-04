# Email/SMS Spam Classifier

## Overview
The **Email/SMS Spam Classifier** is a machine learning project that uses Natural Language Processing (NLP) techniques to classify messages as either **Spam** or **Not Spam**. The project is built with Python and leverages a trained model with TF-IDF vectorization to analyze message content and make predictions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Model Training](#model-training)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Introduction
Spam messages are unsolicited communications often sent in bulk for advertising or malicious purposes. This project aims to provide a tool to detect and classify messages as spam or non-spam, helping users filter unwanted messages efficiently.

## Features
- User-friendly web interface built with Streamlit.
- Input preprocessing including lowercasing, tokenization, removal of stopwords, and stemming.
- Classification model using machine learning algorithms trained with TF-IDF vectorization.
- Instant feedback on message classification (Spam or Not Spam).

## Technologies Used
- **Python**: Programming language.
- **Streamlit**: Framework for creating web applications.
- **NLTK**: Natural Language Toolkit for text preprocessing.
- **scikit-learn**: Machine learning library for training models.
- **Pandas & NumPy**: Libraries for data manipulation and analysis.
- **Pickle**: Used for saving and loading the trained model and vectorizer.

## Project Structure
Project documentation (this file)
SMS-Spam-Detection/
│
├── app.py                 # Streamlit app for user interaction
├── preprocess.py          # Script for data preprocessing
├── trainmodel.py          # Script for training the model
├── spam.csv               # Dataset used for training and testing
├── vectorizer.pkl         # Pickled TF-IDF vectorizer
├── model.pkl              # Pickled trained classification model
├── requirements.txt       # File listing project dependencies
├── README.md              # Project documentation
├── venv/                  # Virtual environment directory
│   └── ...                # Virtual environment files (not typically included in version control)
└── nltk_data/             # Directory for downloaded NLTK resources
    └── ...                # NLTK-specific files



## Installation
Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SMS-Spam-Detection.git
   cd SMS-Spam-Detection
   
2. Create and activate a virtual environment:

  ```python -m venv venv```
   venv/bin/activate  
   # On Windows: 
   ``` venv\Scripts\activate ```
3.Install the required packages:

  ```pip install -r requirements.txt```

  
4. Download necessary NLTK resources:
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
5. How to Use
Run the app.py file to start the Streamlit web application:  
  ```streamlit run app.py```

Enter a message in the text box and click the Predict button.

The app will classify the input as either Spam or Not Spam and display the result on the screen.

### Model Training
The dataset used is `spam.csv`, which includes labeled messages marked as "spam" or "ham" (not spam). The training process involves:
- **Data Cleaning**: Removing unnecessary columns and duplicates
- **Text Preprocessing**: Applying functions to lowercase, tokenize, remove stopwords and punctuation, and stem words.
- **Vectorization**: Using TF-IDF Vectorizer to convert text into numerical features.
- **Model Building**: Training a classifier (e.g., Naive Bayes, Logistic Regression) using scikit-learn.
- **Saving the Model**: Pickling the trained model and vectorizer for future use.

### Results
The trained model effectively distinguishes spam messages from non-spam messages based on content. The app provides immediate feedback after entering a message.

### Future Improvements
- Enhance the user interface with more styling and visual components.
- Integrate additional models for ensemble learning.
- Include a feedback loop for user corrections to improve the model.

### Acknowledgements  
- NLTK for text preprocessing tools.
- scikit-learn for machine learning model implementation.
- The creators of the dataset for providing valuable training data.

### Contact
For any questions or feedback, please reach out: `burathimannu@gmail.com`


### Explanation
- **Detailed Overview**: The `README.md` provides an easy-to-understand overview of the project and its features.
- **Clear Structure**: The `Project Structure` section explains the function of each file in the project.
- **Usage Instructions**: Detailed steps are given for installation and running the app.
- **Future Enhancements**: Suggestions for further development make the project scalable.


