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

The dataset was split into 70% train, 15% validation and 15% test. It was then converted into numerical features using TfidVectorizer, ignoring terms that appear in less than 3 reviews and terms that appear in more than 80% of reviews.





Linear Support Vector Machine was chosen as the baseline model (why). 






## Results 

## Error Analysis

## Discussion 

## Conclusion

## How to Run


