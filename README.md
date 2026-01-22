# Sentiment Analysis of Reviews: Classical vs Transformer Models 

This project explores sentiment analysis of reviews from Yelp. While traditional methods like TF-IDF combined with linear classifiers are fast and interpretable, modern transformer-based models like DistilBERT can capture subtle contextual information, negation, and mixed sentiment. This project investigates the performance differences between these approaches on a real review dataset. 

## Data Description 

Source: https://www.kaggle.com/datasets/ranjulajayarathna/cleaned-yelp-dataset-restaurants-and-reviews?resource=download&select=sampled_yelp_reviews.csv

Text Data: Customer Reviews

Labels: Positive, Negative

Reviews with 4 or 5 stars were labelled as "positive"

Reviews with 1 or 2 stars wre labelled as "negative"

Reviews with neutral sentiment (3 stars) were removed from the dataset

## Exploratory Data Analysis 

There are 8861 total rows in the dataset, with 7019 reviews (79%) being positive and 1842 (21%) being negative. 

On average, negative reviews are longer than positive reviews (51.3 words vs 46.6 words). Negative reviews on average also have more characters than positive reviews (272.8 characters vs 253.2 characters). 

The top 10 words in positive reviews are [('great', 2748), ('place', 2302), ('food', 2191), ('good', 1958), ('service', 1348), ('best', 1046), ('time', 1039), ('love', 980), ('one', 905), ('get', 863)]

The top 10 words in negative reviews are [('food', 622), ('place', 511), ('service', 435), ('good', 337), ('time', 325), ('like', 304), ('one', 299), ('get', 288), ('would', 260), ('order', 250)]

Even after removing standard stopwords (like the, and, is), words like food, place, service are still common but don’t carry strong positive/negative sentiment. Bigrams were added to obtain more contextual information in the text.

Top Positive Bigrams:

[('great food', 193), ('great service', 180), ('love place', 165), ('staff friendly', 158), ('great place', 152), ('food great', 150), ('highly recommend', 146), ('really good', 144), ('new orleans', 132), ('ice cream', 131)]

Top Negative Bigrams:

[('customer service', 68), ('food good', 38), ('tasted like', 37), ('15 minutes', 31), ('don know', 28), ('food service', 24), ('second time', 22), ('20 minutes', 22), ('terrible service', 20), ('long time', 19)]


## Methodology

The dataset was split into 70% train, 15% validation and 15% test. It was then converted into numerical features using TfidVectorizer, ignoring terms that appear in less than 3 reviews and terms that appear in more than 80% of reviews. ngram_range was set at (1,2) to include both unigrams and bigrams. Sublinear_tf scaling (term frequency) was set to True to prevent long reviews from dominating.

```
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),  
    stop_words='english', 
    min_df=3, 
    max_df=0.8, 
    sublinear_tf=True
)
```
Linear Support Vector Machine was chosen as the baseline model (why), with the regularisation parameter C = 1 and class_weight = "balanced" to counteract the imbalanced classes. Performance was evaluated on accuracy, precision, recall, and F1 score.

```
svm = LinearSVC(
    C=1.0,
    class_weight="balanced"  
)
```
DistilBert was selected as the transformer model. A pre-trained DistilBERT model (distilbert-base-uncased) for sentiment classification was selected.

```
model_name = "distilbert-base-uncased"
```

Reviews were tokenized using DistilBertTokenizerFast, padded and truncated to a maximum length of 128 tokens.

```
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# DistilBERT for classification (num_labels = 2 for binary)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

```

The model was trained on our dataset for 3 epochs using the Hugging Face Trainer API. Similar to the baseline model, performance was evaluated on the test set using accuracy, precision, recall, and F1 score.

```

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
```

## Results 

### Final Test - Linear SVM

The final model maintained strong performance on the unseen test set, achieving an accuracy of approximately 89% and a macro-averaged F1 score of around 0.83.
Negative class F1 score was 0.74, while positive class F1 score was 0.93. There was a slight bias towards the positive class due to class imbalance.

### Final Test - DistilBert

DistilBERT achieved 93% accuracy on the test set, a 10% improvement on the Linear SVM model. It also achieved an F1 score of 0.84 for negative reviews and 0.96 for positive reviews. Compared to the Linear SVM baseline, DistilBERT improved detection of negative sentiment while maintaining high performance on positive reviews.

    
## Error Analysis

False Negatives and False Positives were analysed to understand the errors made by the models. 

### False Negatives
1) Label Noise - These were reviews where the review was actually negative however it had the wrong label.

"new owners turned it into a double threat bad food and bad service shame i loved this place"

2) Mixed Sentiment - These errors were due to long reviews with both positive and negative words, with the model choosing to focus on negative words

"i have been renting a home from principal property management for years although we initially had some bumps and misunderstandings it has overall been a very positive experience i have had some pretty major issues at my home with the ac and refrigerator going out and all was resolved"

3) Keyword Confusion/ Context

"i have a feeling people only write reviews about party supply places if they have a bad experience so let me change that up jumpmaxx is a great company.."


### False Positives 

1) Sarcasm/Irony

"surrey's is allegedly the best breakfast in new orleans yelpers talk about it like it will change your life i do not understand and i never will i have eaten at surrey's three times in the past year and each time my order has been borderline"

2) Mixed Sentiment

"i just moved in nearby so my friends and i stopped in la va for a mid moving rest lunch we were a little disappointed but not entirely let down the place itself has a good vibe so i would consider going back to hang out with a coffee we ordered sandwiches because we were starving after moving"

3) Negation Missed

"the reviews seemed promising maybe they had an off night but the food here was not good"



## Conclusion

## How to Run


