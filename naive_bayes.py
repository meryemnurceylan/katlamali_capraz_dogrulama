import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

# Kelimelerin köklerini belirlemek için gereken fonksiyon
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return None

# İşlenmiş verileri içeren CSV dosyasını oku
data = pd.read_csv(r"C:\Users\merye\OneDrive\Masaüstü\derin öğrenme\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Duygu değerlerini hesaplamak için gereken fonksiyon
def sentiment_score(rating):
    if rating > 3:
        return 1  # Olumlu
    elif rating < 3:
        return -1  # Olumsuz
    else:
        return 0  # Tarafsız

# İşlenmiş yorumları saklamak için bir liste oluştur
processed_reviews = []

# Stopword'lerin listesini oluştur
stop_words = set(stopwords.words('english'))

# Lemmatizer oluştur
lemmatizer = WordNetLemmatizer()

# Yorumları önişleme
for index, row in data.iterrows():
    if isinstance(row['reviews.text'], str):  # Metin verisi olduğundan emin ol
        review = row['reviews.text']
        # Belirteçleme
        tokens = nltk.word_tokenize(review)
        # Kök belirleme ve etiketleme (lemmatization ve pos tagging)
        tagged_tokens = nltk.pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tagged_tokens if get_wordnet_pos(tag)]
        # Stopword'leri çıkar
        filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]
        # Tüm harfleri küçük harfe dönüştür
        lowercased_tokens = [word.lower() for word in filtered_tokens]
        processed_reviews.append(" ".join(lowercased_tokens))

# İşlenmiş metinleri 'processed_text' adında bir sütun olarak DataFrame'e ekle
data['processed_text'] = processed_reviews

# Duygu sütununu oluştur
data['sentiment'] = data['reviews.rating'].apply(sentiment_score)

# Özellik ve hedef sütunları seç
X = data['processed_text']  # İşlenmiş metin sütunu
y = data['sentiment']        # Duygu sütunu

# TF-IDF dönüşümü
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Sınıflandırıcıları tanımla
classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Çapraz doğrulama skorlarını depolamak için bir DataFrame oluştur
cv_scores = pd.DataFrame(index=classifiers.keys(), columns=['Mean Accuracy'])

# Her sınıflandırıcı için çapraz doğrulama yap
for clf_name, clf in classifiers.items():
    scores = cross_val_score(clf, X_tfidf, y, cv=10, scoring='accuracy')
    cv_scores.loc[clf_name, 'Mean Accuracy'] = np.mean(scores)

print(cv_scores)
