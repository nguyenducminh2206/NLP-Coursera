import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

fig = plt.figure()
labels = 'Positive', 'Negative'
sizes = [len(positive_tweets), len(negative_tweets)]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
# plt.show()

# Example of a positive tweet
# print(positive_tweets[random.randint(0,5000)])

# Example of a negative tweet
# print(negative_tweets[random.randint(0,5000)])

tweet = positive_tweets[2277]
print(tweet)

# Remove hyperlinks, Twitter marks and styles #

# Remove old style retweet text 'RT'
tweet2 = re.sub(r'RT[\s]+', '', tweet)

# Remove hyperlinks
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)

# Remove hashtags
tweet2 = re.sub(r'#', '', tweet2)
print(tweet2)
print()

# Tokenize the string #
tokenizer = TweetTokenizer(
    preserve_case=False, strip_handles=True, reduce_len=True)

tweet_tokens = tokenizer.tokenize(tweet2)

print(tweet_tokens)
print()

# Import the english stop words list from NLTK
stopwords_english = stopwords.words('english')

tweet_clean = []
for word in tweet_tokens:
    if word not in stopwords_english and word not in string.punctuation:
        tweet_clean.append(word)

print('remove stopwords and punctuation:', tweet_clean)
print()

# Stemming #
stemmer = PorterStemmer()

tweets_stem = []

for word in tweet_clean:
    stem_word = stemmer.stem(word)
    tweets_stem.append(stem_word)

print('stem word:', tweets_stem)
