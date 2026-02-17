import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import urllib.request
import re

DATA_URL = "https://www.gutenberg.org/files/913/913-0.txt"

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find("stopwords")
    nltk.data.find("punkt")
except LookupError:
    nltk.download("stopwords")
    nltk.download('punkt_tab')
    nltk.download("wordnet")
    nltk.download('punkt')

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

resource = urllib.request.urlopen(url=DATA_URL, timeout=5)
charset = resource.headers.get_content_charset()
print("Charset", charset)
raw_text = resource.read()

if charset:
  raw_text = raw_text.decode(resource.headers.get_content_charset())
else:
  raw_text = raw_text.decode("utf-8")

raw_text[:100]

# removing the book ending
clean_pattern = re.compile("End of the Project Gutenberg EBook.*")
cleaner_text =  re.sub(clean_pattern, "", raw_text.replace("\n", " ").replace("\r", " "))
cleaner_text[-100:]

tokens =  word_tokenize(cleaner_text)
print("the number of tokens", len(tokens))

# removing incorrect words, taking only lemmas
lemmas = list(filter(lambda t: t.isalpha(), [lemmatizer.lemmatize(lemma) for lemma in tokens]))

freq_dist = FreqDist(lemmas)
STOPWORDS = set(stopwords.words("english"))
top_freq_lemmas = sorted(freq_dist.items(), key=lambda x: x[1], reverse=True)[:50]
number_intersection_keys = len(set(map(lambda x: x[0], top_freq_lemmas)) & STOPWORDS)
print(number_intersection_keys)
frequent_lemmas = {lemma: count for lemma, count in freq_dist.items() if count > 20}
print(len(frequent_lemmas))