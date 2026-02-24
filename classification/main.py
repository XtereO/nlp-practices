import re

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import stem
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

df = pd.read_csv('3_data.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df = df.rename(columns = {'v1': 'label', 'v2': 'text'})
df.head()

df = df.drop_duplicates('text')

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

nltk.download('stopwords')
nltk.download('punkt')
stemmer = SnowballStemmer("english")

stemmer = stem.SnowballStemmer('english')
stopwords = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text)
    words = pd.Series([text]).str.split()[0] # word_tokenize(text) #
    words = [word.lower() for word in words]
    filtered_words = [word for word in words if word not in stopwords]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    tokens = stemmed_words
    
    '''
    corrected_tokens = []
    i = 0
    while i < len(tokens):
      if tokens[i] == "gon" and i < len(tokens) - 1 and tokens[i + 1] == "na":
        corrected_tokens.append("gonna")  # Combine 'gon' and 'na'
        i += 2  # Skip the next token
      else:
        corrected_tokens.append(tokens[i])
        i += 1
    '''
        
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

#assert preprocess("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.") == "im gonna home soon dont want talk stuff anymor tonight k ive cri enough today"
#assert preprocess("Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...") == "go jurong point crazi avail bugi n great world la e buffet cine got amor wat"


df['text'] = df['text'].apply(preprocess)

y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.35, random_state=34)

# exctract features from the texts
vectorizer = TfidfVectorizer(decode_error='ignore')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LinearSVC(random_state = 34, C = 1.3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions, digits=3))

pred_txt = ["Excellent collection of articles and speeches.",
            "You are cordially invited to the 2021 International Conference on Advances in Digital Science (ICADS 2021), to be held at Salvador, Brazil, 19 – 21 February 2021, is an international forum for researchers and practitioners to present and discuss the most recent innovations. Ответ {$а} Вопрос 1",
            "URGENT! We are trying to contact U.Todays draw shows that you have won a 2000 prize GUARANTEED. Call 090 5809 4507 from a landline. Claim 3030. Valid 12hrs only.",
            "Machine factory for sale. Low price. 2 hectare property, 150,000 square feet production floor, 500 machine tools installed."]
pred_txt = [preprocess(txt) for txt in pred_txt]
pred_txt = vectorizer.transform(pred_txt)

print(model.predict(pred_txt))
