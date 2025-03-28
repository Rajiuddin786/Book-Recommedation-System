import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


books=pd.read_excel('books.xlsx')


def weighted_rating(row, m=100, C=books['average_rating'].mean()):
    v = row['ratings_count']  # Number of ratings
    R = row['average_rating']  # Average rating
    return (v / (v + m) * R) + (m / (m + v) * C)

books['weighted_rating'] = books.apply(lambda x: weighted_rating(x), axis=1)

books['review_weight'] = books['text_reviews_count'] / books['text_reviews_count'].max()
books['final_score'] = books['weighted_rating'] * 0.8 + books['review_weight'] * 0.2

books = books.groupby('title').apply(lambda x: x.loc[x['ratings_count'].idxmax()])
books.reset_index(drop=True, inplace=True)


# Combine text features
books['combined_features'] = books['authors'] + " " + books['publisher'] + " " + books['language_code']

# Apply TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined_features'])

# Compute similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend similar books
def recommend_similar_books(book_title, top_n=5):
    idx = books.index[books['title'] == book_title]
    if not idx.empty:
        idx=idx[0]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        book_indices = [i[0] for i in scores]
        return books.iloc[book_indices][['title', 'authors', 'publisher','final_score']]
    else:
        return pd.DataFrame()


# Prepare data for surprise
reader = Reader(rating_scale=(1, 5))
books['user_id']=0
data = Dataset.load_from_df(books[['user_id','bookID', 'final_score']], reader)

# Train SVD model
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)

# Predict rating for a book
def predict_book_rating(book_id):
    return svd.predict(book_id, 0).est  # Using a dummy user_id (0)

def hybrid_recommend(book_title, top_n=5):
    similar_books = recommend_similar_books(book_title, top_n)
    if similar_books.empty:
        return []
    else:
        similar_books['predicted_rating'] = similar_books['title'].apply(
            lambda title: predict_book_rating(books.index[books['title'] == title][0])
        )
        return similar_books.sort_values(by='predicted_rating', ascending=False)
print(hybrid_recommend("Dubliners").title)