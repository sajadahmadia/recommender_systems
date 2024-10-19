from distutils.command import clean
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity


#correct and lowercase tags function
def correct_and_lower(tags):
    if isinstance(tags, list):
        return [str(tag).lower() for tag in tags]
    return tags

#combine and deduplicate descriptions and tags
def combine_and_deduplicate(field1, field2):
    combined_field = field1 + " " + field2
    deduplicated_field = ' '.join(set(combined_field.split()))
    return deduplicated_field

def combine_and_deduplicate_lists_corrected(list1, list2):
    list1 = list1 if isinstance(list1, list) else []
    list2 = list2 if isinstance(list2, list) else []
    combined_list = list(set(list1 + list2))
    return ' '.join(combined_list)

#one-hot encoding and Jaccard similarity calculation for 'rating'
def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    row_sums = intersection.diagonal()
    unions = row_sums[:, None] + row_sums - intersection
    return intersection / unions


def get_liked_movies_from_user_data(df_user, user_id, movie_rating = 4, watched_threshold=85):
    watched_movies_df = df_user[(df_user['user_id'] == user_id) & 
                                (df_user['watched_percentage'] >= watched_threshold)&
                                (df_user['rating'] >= movie_rating)]
    return watched_movies_df['title'].tolist()


def recommend_based_on_similarity(input_titles, df, combined_similarity):
       # Find the indices of the input titles in the dataset
    input_indices = [i for i, title in df['title'].items() if title in input_titles['title'].values]
   
    # input_indices = [df.index[df['title'] == title].tolist()[0] for title in input_titles if title in df['title'].values]
    if not input_indices:
        return {}, "No input titles found in the dataset."
    
    # Aggregate similarity scores across all input titles
    aggregate_similarity_scores = np.mean(combined_similarity[input_indices, :], axis=0)
    
    # Normalize the aggregate similarity scores between 0 and 1
    min_score = np.min(aggregate_similarity_scores)
    max_score = np.max(aggregate_similarity_scores)
    normalized_scores = (aggregate_similarity_scores - min_score) / (max_score - min_score)
    
    # Create a dictionary to hold movie titles and their normalized similarity scores
    movie_scores = {df.loc[i, 'title']: score for i, score in enumerate(normalized_scores) if i not in input_indices}
    
    # Order the dictionary by similarity scores (descending, for most similar movies first)
    ordered_movie_scores = dict(sorted(movie_scores.items(), key=lambda item: item[1], reverse=True))
    
    return ordered_movie_scores


def cleaner(show_df):
    #drop any duplicate entries to ensure only movies remain
    big_df = show_df.drop_duplicates(subset=['title'], keep='first')

    big_df['tags'] = big_df['tags'].apply(correct_and_lower)
    big_df['tags2'] = big_df['tags2'].apply(correct_and_lower)

    big_df['deduplicated_description'] = big_df.apply(lambda x: combine_and_deduplicate(x['description'], x['description2']), axis=1)
    big_df['deduplicated_tags'] = big_df.apply(lambda x: combine_and_deduplicate_lists_corrected(x['tags'], x['tags2']), axis=1)

    #reset the index of the dataframe to ensure alignment after preprocessing
    big_df = big_df.reset_index(drop=True)

    return big_df

# combined similarity function
def combined_coverage_similarity(show_df):
    #TF-IDF Vectorizer initialization and matrix generation
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_descriptions = tfidf_vectorizer.fit_transform(show_df['deduplicated_description'])
    tfidf_matrix_tags = tfidf_vectorizer.fit_transform(show_df['deduplicated_tags'])

    #cosine similarity description
    cosine_sim_descriptions = cosine_similarity(tfidf_matrix_descriptions, tfidf_matrix_descriptions)
    cosine_sim_descriptions_normalized = cosine_sim_descriptions / np.max(cosine_sim_descriptions)

    #cosine similarity tags
    cosine_sim_tags = cosine_similarity(tfidf_matrix_tags, tfidf_matrix_tags)
    cosine_sim_tags_normalized = cosine_sim_tags / np.max(cosine_sim_tags)

    #one-hot encoding and Jaccard similarity calculation for 'rating'
    ohe_encoder = OneHotEncoder(sparse_output=False)
    ratings_encoded = ohe_encoder.fit_transform(show_df[['rating']])
    jaccard_sim_ratings = jaccard_similarity(ratings_encoded)
    jaccard_sim_ratings_normalized = jaccard_sim_ratings / np.max(jaccard_sim_ratings)

    #combine and normalize similarity scores
    combined_similarity = (cosine_sim_descriptions_normalized + cosine_sim_tags_normalized + jaccard_sim_ratings_normalized) / 3

    return combined_similarity

# Initialize global variables for tracking recommended items and recommendation counts
recommendation_counts = {}
recommended_items_set = set()


# Define the function to adjust scores for coverage
def adjust_scores_for_coverage(sim_scores, recommendation_counts):
    adjusted_scores = []
    for idx, score in sim_scores:
        penalty = recommendation_counts.get(idx, 0)
        adjusted_score = score * (1 / (1 + penalty))
        adjusted_scores.append((idx, adjusted_score))
    return adjusted_scores


# Function to recommend with individual genre selection per recommendation
def recommend_with_individual_genre_selection(input_titles, df, combined_similarity_refined, already_recommended, total_recommendations):
    recommended_titles = []
    selected_genres = []
    
    for input_title in input_titles:
        if input_title not in df['title'].values:
            recommended_titles.append("Movie title not found.")
            selected_genres.append(None)
            continue
        
        input_idx = df.index[df['title'] == input_title].tolist()[0]
        unique_genres = df['genre'].dropna().unique()
        
        for _ in range(total_recommendations):
            selected_genre = np.random.choice(unique_genres)
            selected_genres.append(selected_genre)
            
            genre_df = df[(df['genre'] == selected_genre) & (~df['title'].isin(already_recommended))].reset_index()
            original_indices = genre_df['index'].values
            
            genre_similarities = combined_similarity_refined[input_idx, original_indices]
            
            sim_scores = [(idx, genre_similarities[i]) for i, idx in enumerate(original_indices)]
            adjusted_scores = adjust_scores_for_coverage(sim_scores, recommendation_counts)
            adjusted_scores = sorted(adjusted_scores, key=lambda x: x[1], reverse=True)
            
            
            for i, _ in adjusted_scores[:total_recommendations]:
                if i != input_idx and len(recommended_titles) < total_recommendations:
                    recommended_items_set.add(df.loc[i, 'title'])
                    recommendation_counts[i] = recommendation_counts.get(i, 0) + 1
                    recommended_titles.append(df.loc[i, 'title'])

    return recommended_titles, selected_genres

# New function to calculate coverage
def catalog_coverage(df):
    total_items = len(df['title'].unique())
    coverage_rate = len(recommended_items_set) / total_items
    return coverage_rate


# New function to calculate Gini index
def calculate_gini_index(proportions):
    """
    Calculate the Gini index for a list of proportions.
    
    Parameters:
    - proportions: A list of proportions p(i) for each item i
    
    Returns:
    - gini_index: The calculated Gini index
    """
    n = len(proportions)
    # Ensure the list is sorted by increasing p(i)
    sorted_proportions = np.sort(proportions)
    # Calculate the Gini index using the provided formula
    gini_index = (2 / (n - 1)) * np.sum((np.arange(1, n+1) - (n + 1) / 2) * sorted_proportions)
    gini_index = 1 - gini_index  # According to the formula provided, this should be subtracted from 1
    return gini_index


# New function to calculate Shannon entropy
def calculate_shannon_entropy(recommendation_counts):
    counts = np.array(list(recommendation_counts.values()), dtype=float)
    total = counts.sum()
    if total > 0:
        probabilities = counts / total
        nonzero_probabilities = probabilities[probabilities > 0]
        shannon_entropy = -np.sum(nonzero_probabilities * np.log2(nonzero_probabilities))
    else:
        shannon_entropy = 0
    return shannon_entropy
