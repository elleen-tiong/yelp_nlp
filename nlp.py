#dataset source 
# https://www.kaggle.com/datasets/ranjulajayarathna/cleaned-yelp-dataset-restaurants-and-reviews?resource=download&select=sampled_yelp_reviews.csv

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from transformers import TrainingArguments, Trainer, DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch


df = pd.read_csv('/Users/elleentiong/Desktop/NLP_Project/sampled_yelp_reviews.csv')

# reviews with neutral sentiment (3 stars) were removed
#reviews with 4 or 5 stars were labeled as positive sentiment
#reviews with 1 or 2 stars were labeled as negative sentiment
df = df[df["stars"] != 3]
df["sentiment"] = np.where(df["stars"] >= 4, "positive", "negative")
#df.to_csv("/Users/elleentiong/Desktop/NLP_Project/data_no_neutral.csv", index=False)
label_map = {"negative": 0, "positive": 1}
#EXPLORATORY DATA ANALYSIS 
#Dataset size + class balance
#Text length analysis
#Top words per sentiment
#Top bigrams per sentiment

#Dataset size + class balance
print("Total rows:", df.shape[0]) #8861 rows 
#print(df['sentiment'].value_counts()) #7019 positive sentiments, 1842 negative sentiments
#print(df['sentiment'].value_counts(normalize=True) * 100) #posiritive sentiments 79%, negative sentiments 21%

#TEXT LENGTH ANALYSIS 

#average number of words per review
df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
avg_words = df.groupby('sentiment')['word_count'].mean()
#print(avg_words) #on average, negative reviews are longer than positive reviews (51.3 words vs 46.6 words)

#average number of characters per review
df['char_count'] = df['text_clean'].apply(lambda x: len(str(x)))
avg_chars = df.groupby('sentiment')['char_count'].mean()
#print(avg_chars) #on average, negative reviews have more characters than positive reviews (272.8 characters vs 253.2 characters)

#summary table for text length analysis 
summary_table = df.groupby('sentiment').agg({
    'word_count': 'mean',
    'char_count': 'mean'
}).reset_index()

# Rename for readability
summary_table['sentiment'] = summary_table['sentiment'].map({0: 'Negative', 1: 'Positive'})
summary_table.rename(columns={
    'word_count': 'Mean Word Count',
    'char_count': 'Mean Character Count'
}, inplace=True)

#print(summary_table)

#MOST COMMON WORDS 

#download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

pos_reviews = df[df['sentiment'] == "positive"]['text_clean']
neg_reviews = df[df['sentiment'] == "negative"]['text_clean']

#print(pos_reviews.head())

def preprocess_text(text_clean):
    # Convert to lowercase
    text_clean = text_clean.lower()
    # Remove punctuation
    text_clean = text_clean.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    words = [w for w in text_clean.split() if w not in stop_words]
    return words

# Flatten all words into a single list per class
pos_words = [word for review in pos_reviews for word in preprocess_text(review)]
neg_words = [word for review in neg_reviews for word in preprocess_text(review)]

# Count word frequencies
pos_counts = Counter(pos_words)
neg_counts = Counter(neg_words)

print("Top 10 words in positive reviews:")
print(pos_counts.most_common(10))

print("\nTop 10 words in negative reviews:")
print(neg_counts.most_common(10))
# top words in negative reviews are food, place, service, good, time, like, one, get, would, order
#Even after removing standard stopwords (like the, and, is), words like food, place, service are still common but don’t carry strong positive/negative sentiment.
#moving onto bigrams to see if we can capture more context

#N-GRAMS ANALYSIS 

def get_top_ngrams(text_series, ngram_range=(2,2), n=10):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(text_series)
    counts = X.sum(axis=0)
    ngram_counts = [(ngram, int(counts[0, idx])) for ngram, idx in vectorizer.vocabulary_.items()]
    ngram_counts = sorted(ngram_counts, key=lambda x: x[1], reverse=True)
    return ngram_counts[:n]

# bigrams
print(get_top_ngrams(pos_reviews, ngram_range=(2,2), n=10))
print(get_top_ngrams(neg_reviews, ngram_range=(2,2), n=10))

#Top Positive Bigrams:
#[('great food', 193), ('great service', 180), ('love place', 165), ('staff friendly', 158), ('great place', 152), ('food great', 150), ('highly recommend', 146), ('really good', 144), ('new orleans', 132), ('ice cream', 131)]

#Top Negative Bigrams:
#[('customer service', 68), ('food good', 38), ('tasted like', 37), ('15 minutes', 31), ('don know', 28), ('food service', 24), ('second time', 22), ('20 minutes', 22), ('terrible service', 20), ('long time', 19)]


#TRAIN-VALID-TEST SPLIT 

#70%train, 15% validation, 15% test

df['text_clean'] = (
    df['text_clean']
      .str.lower()  # lowercase
      .str.replace(r"[^a-zA-Z\s']", " ", regex=True)  # keep letters, spaces, apostrophes
      .str.replace(r"\s+", " ", regex=True)  # replace multiple spaces with single space
      .str.strip()  # remove leading/trailing spaces
)

df['text_clean'] = df['text_clean'].apply(contractions.fix)

#print(df['text_clean'].head())


X = df["text_clean"]
y = df["sentiment"]

# #train vs temp (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,        # 30% goes to val + test
    random_state=42,
    stratify=y
)

#validation vs test 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

y_train = [label_map[y] for y in y_train]
y_val   = [label_map[y] for y in y_val]
y_test  = [label_map[y] for y in y_test]

# print("Train:", y_train.value_counts(normalize=True))
# print("Val:", y_val.value_counts(normalize=True))
# print("Test:", y_test.value_counts(normalize=True))

#Baseline Model 

#step 1: convert text to numerical features using tfidf 
#step 2: train a simple classifier
#step 3: evaluate on validation and test set 

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),  # unigrams + bigrams
    stop_words='english',  # removes common English stopwords
    min_df=3, #ignore terms that appear in less than 3 reviews
    max_df=0.8, #ignore terms that appear in more than 80% of reviews
    sublinear_tf=True # apply sublinear tf scaling to prevent long reviews from dominating
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)

#training a base model (linear SVM)

svm = LinearSVC(
    C=1.0,
    class_weight="balanced"  # optional, good if classes are imbalanced
)

svm.fit(X_train_tfidf, y_train)

#validating the model 

y_val_pred = svm.predict(X_val_tfidf)

#print("Validation Accuracy:", accuracy_score(y_val, y_val_pred)) #0.898 accuracy on validation set

#print(classification_report(y_val, y_val_pred))
#negative class
#precision = 0.73 - When the model predicts negative, it’s correct 73% of the time
#recall = 0.80 - The model identifies 80% of all actual negative reviews

#positive class 
#precision = 0.95, recall = 0.92, decent performance 

#macro f1 = 0.85
#there is a slight bias towards the positive class due to class imbalance

#Despite class imbalance, the model achieved a macro-averaged F1 score of 0.85, indicating robust performance across both classes. 
#Performance on the minority (negative) class remained strong with an F1 of 0.77.”

#FINAL MODEL 

final_svm = LinearSVC(
    C=1.0,
    class_weight="balanced"
)

final_svm.fit(X_train_tfidf, y_train)

y_test_pred = final_svm.predict(X_test_tfidf)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

#test results
#Accuracy ≈ 0.89
#Macro F1 ≈ 0.83
#Negative F1 ≈ 0.74

#The final model maintained strong performance on the unseen test set, achieving an accuracy of approximately 89% and a macro-averaged F1 score of around 0.83.

#Negative (minority class)

#Precision = 0.74
#Recall = 0.74
#F1 = 0.74
#Balanced precision/recall

#Positive (majority class)
#Precision = 0.93
#Recall = 0.93
#F1 = 0.93

#A Linear Support Vector Machine trained on TF-IDF unigram and bigram features achieved an accuracy of 89.0% and 
# a macro-averaged F1 score of 0.83 on the held-out test set. 
# Performance remained strong across both classes despite class imbalance.”


#ERROR ANALYSIS 

df_test = pd.DataFrame({
    "text": X_test,
    "true": y_test,
    "pred": y_test_pred
})


# False positives: model predicts positive, true is negative
false_positives = df_test[(df_test["pred"] == "positive") & (df_test["true"] == "negative")]

# False negatives: model predicts negative, true is positive
false_negatives = df_test[(df_test["pred"] == "negative") & (df_test["true"] == "positive")]

#sarcasm/mixed sentiment 
#false_positives["text"].to_csv("/Users/elleentiong/Desktop/NLP_Project/false_positives.txt", index=False)
#false_negatives["text"].to_csv("/Users/elleentiong/Desktop/NLP_Project/false_negatives.txt", index=False)

#ADVANCED MODEL: DISTILBERT 

# Choose the pretrained model
model_name = "distilbert-base-uncased"

# Tokenizer converts text to tokens
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# DistilBERT for classification (num_labels = 2 for binary)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

train_texts = X_train.fillna("").tolist() # your raw train text
val_texts = X_val.fillna("").tolist()
test_texts = X_test.fillna("").tolist()

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ReviewsDataset(train_encodings, y_train)
val_dataset = ReviewsDataset(val_encodings, y_val)
test_dataset = ReviewsDataset(test_encodings, y_test)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Evaluate on validation set
metrics = trainer.evaluate()
print(metrics)
preds = trainer.predict(test_dataset)
y_pred_bert = preds.predictions.argmax(axis=1)

print("DistilBERT Accuracy:", accuracy_score(y_test, y_pred_bert))
print(classification_report(y_test, y_pred_bert, target_names=["negative", "positive"]))