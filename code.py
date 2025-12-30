import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

courses_df = pd.DataFrame(courses_data)
ratings_df = pd.DataFrame(ratings_data)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(courses_df["skills"])

def content_based_recommend(user_interest, user_level):
    user_vec = tfidf.transform([user_interest])
    similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
    courses_df["content_score"] = similarity
    level_filtered = courses_df[courses_df["level"] == user_level]
     return level_filtered

def collaborative_filtering(user_id):
    user_course_matrix = ratings_df.pivot(index="user_id", columns="course_id", values="rating").fillna(0)  
    similarity = cosine_similarity(user_course_matrix)
    sim_df = pd.DataFrame(similarity, index=user_course_matrix.index, columns=user_course_matrix.index)
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:3].index
    recommendations = ratings_df[ratings_df["user_id"].isin(similar_users)]
    return recommendations.groupby("course_id")["rating"].mean()

def hybrid_recommendation(user_id, interest, level, top_n=3):
    cb = content_based_recommend(interest, level)
  
    if user_id in ratings_df["user_id"].values:
        cf_scores = collaborative_filtering(user_id)
        cb["cf_score"] = cb["course_id"].map(cf_scores)
        cb["cf_score"] = cb["cf_score"].fillna(0)
    else:
        cb["cf_score"] = 0  # Cold start
    
    cb["popularity_score"] = cb["popularity"] / 100
    
    cb["final_score"] = (
        0.5 * cb["content_score"] +
        0.3 * cb["cf_score"] +
        0.2 * cb["popularity_score"]
    )   
    recommendations = cb.sort_values("final_score", ascending=False).head(top_n)
    return recommendations[["title", "level", "final_score"]]

user_id = int(input("Enter User ID: "))
interest = input("Enter your interests: ")
level = input("Enter skill level (Beginner/Intermediate/Advanced): ")

result = hybrid_recommendation(user_id, interest, level)

print("\nRecommended Courses:")
print(result.to_string(index=False))

