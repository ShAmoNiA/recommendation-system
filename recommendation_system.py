import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('data.csv')


df['description'] = df['description'].fillna('')
df['features'] = df['features'].fillna('')
df['text'] = df['description'] + ' ' + df['features']


train_data, test_data = train_test_split(df, test_size=0.2)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_train = tfidf_vectorizer.fit_transform(train_data['text'])
tfidf_test = tfidf_vectorizer.transform(test_data['text'])


similarity_matrix = cosine_similarity(tfidf_train)


def recommend_items(item_id, similarity_matrix, train_data, num_items=10):
    item_index = train_data[train_data['item_id'] == item_id].index[0]
    item_similarity_scores = list(enumerate(similarity_matrix[item_index]))
    sorted_scores = sorted(item_similarity_scores, key=lambda x:x[1], reverse=True)
    top_items = [i[0] for i in sorted_scores[1:num_items+1]]
    return train_data.iloc[top_items]

item_id = 12345
recommended_items = recommend_items(item_id, similarity_matrix, train_data, num_items=10)
print(recommended_items)