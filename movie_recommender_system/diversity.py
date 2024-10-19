# importing libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
nltk.download('wordnet')
nltk.download('stopwords')


def movie_recommender(user_data, user_id, num_neighbors, implicit_feedback_weight=0.5):
    """_summary_

    Args:
        user_data_path (string): url or path to the user data 
        user (string): a unique user id from the user data
        num_neighbors (int): number of neighbors used in the KNN model
        implicit_feedback_weight (float, optional): the weight for the implicit feedback(here, percentage watched). Defaults to 0.5.

    Returns:
        pd.Series: showing the predicted and normalized(between 0 to 1) rate of the user for each movie in the user data dataset
    """

    # assuming each user has only one final rating for each movie title
    df = user_data.drop_duplicates(subset=['title', 'user_id'])

    # creating the item-user matrices and normalizing them
    df_rating = df.pivot(index='title', columns='user_id',
                         values='rating').fillna(0) / 5.0

    df_watched = df.pivot(index='title', columns='user_id',
                          values='watched_percentage').fillna(0)/100.0

    # combining the implicit and explicit feedbacks
    df = (1-implicit_feedback_weight) * df_rating + \
        implicit_feedback_weight * df_watched
    df1 = df.copy()

    number_neighbors = num_neighbors

    # finding the n nearest users to the target user
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(
        df.values, n_neighbors=number_neighbors)

    user_index = df.columns.tolist().index(user_id)

    for m, t in list(enumerate(df.index)):
        if df.iloc[m, user_index] == 0:
            sim_movies = indices[m].tolist()
            movie_distances = distances[m].tolist()

            if m in sim_movies:
                id_movie = sim_movies.index(m)
                sim_movies.remove(m)
                movie_distances.pop(id_movie)

            else:
                sim_movies = sim_movies[:number_neighbors-1]
                movie_distances = movie_distances[:number_neighbors-1]

            movie_similarity = [1-x for x in movie_distances]
            movie_similarity_copy = movie_similarity.copy()
            nominator = 0

            for s in range(0, len(movie_similarity)):
                if df.iloc[sim_movies[s], user_index] == 0:
                    if len(movie_similarity_copy) == (number_neighbors - 1):
                        movie_similarity_copy.pop(s)

                    else:
                        movie_similarity_copy.pop(
                            s-(len(movie_similarity)-len(movie_similarity_copy)))

                else:
                    nominator = nominator + \
                        movie_similarity[s]*df.iloc[sim_movies[s], user_index]

            if len(movie_similarity_copy) > 0:
                if sum(movie_similarity_copy) > 0:
                    predicted_r = nominator/sum(movie_similarity_copy)

                else:
                    predicted_r = 0

            else:
                predicted_r = 0

            df1.iloc[m, user_index] = predicted_r
    return df.loc[:, user_id], df.loc[:, user_id].to_dict()


# to read the items metadata file
def data_reader(given_data, common_columns=['title', 'description']):
    """_summary_

    Args:
        path (string): path to the items(movies) combined file with pk suffix
        common_columns (list, optional): only these columns will be returned. Defaults to ['title', 'description'].

    Returns:
        pd.DataFrame: returns a dataframe with 3 columns, two are the common_columns
        and a new one, the concatnation of the common_columns, named as feature
    """
    data = given_data
    data = data[common_columns].drop_duplicates(subset=['title'])
    data = data.reset_index(drop=True)
    data['feature'] = data.apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1)
    return data


# preprocess data
def preprocessing(dataframe, column_name='feature'):
    """_summary_

    Args:
        dataframe (pandas dataframe): dataframe containing at least one string column
        column_name (str, optional): the target column to be preproccesed. Defaults to 'feature'.

    Raises:
        ValueError: in case of missing column name, raises error 

    Returns:
        pd.DataFrame: a dataframe in which the target column is cleaned and lemmatized 
    """
    lemmatizer = WordNetLemmatizer()
    pattern = r'[^a-zA-Z0-9\s]'

    def preprocess_text(text):
        text = re.sub(pattern, '', text)
        lemmatized_text = ' '.join(
            [lemmatizer.lemmatize(word.lower()) for word in text.split()])
        return lemmatized_text

        # Check if the specified column exists
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

        # Apply the preprocessing to the specified column
    dataframe[column_name] = dataframe[column_name].astype(
        str).apply(preprocess_text)

    return dataframe


def similarity_matrix(dataframe, column):
    """_summary_

    Args:
        dataframe (pd.DataFrame): pandas data frame containing the "column" and n items
        column (_type_): target column to measure consine similarity based on it

    Raises:
        ValueError: in case of missing column name 

    Returns:
        numpy array: containing the pairwise similarity values n*n
    """
    try:
        tfidf_vectorizer_combined = TfidfVectorizer(stop_words='english')
        tfidf_matrix_combined = tfidf_vectorizer_combined.fit_transform(
            dataframe[column])
        cosine_sim_combined = cosine_similarity(
            tfidf_matrix_combined, tfidf_matrix_combined)
        return cosine_sim_combined
    except Exception as e:
        raise ValueError(
            f"An error occured while computing the similarity matrix: {e}")


def diversity_score(user_id, user_data, items_df, cosine_matrix, item_col='title', n_neighbors=3, recoms_length=10):
    """_summary_

    Args:
        user (string): user_id from the users' data to get recommendations 
        user_data_path (string): path to the users data file
        items_df (pd.DataFrame): the dataframe containing the items metadata 
        item_col (str, optional): the column containing the items unique values in the items_df. Defaults to 'title'.
        n_neighbors (int, optional): number of neighbors similar to the user. Defaults to 3.
        recoms_length (int, optional): the number of recommended items to the user. Defaults to 10.

    Raises:
        ValueError: in case of missing values 

    Returns:
        dict: keys will be movie titles, values are dissimilarity scores
    """
    try:
        initial_recoms_series, initial_recomes_dict = movie_recommender(
            user_data, user_id, 3, implicit_feedback_weight=0.5)
        top_n_recoms = list(initial_recoms_series.sort_values(
            ascending=False).index[:recoms_length])
        top_n_recoms_indices = items_df[items_df[item_col].isin(
            top_n_recoms)].index
        sim_score_indices = cosine_matrix[top_n_recoms_indices]
        movie_to_list_avg_dissimilarity = 1 - \
            (np.sum(sim_score_indices, axis=0) / 10)
        dissimilarity_scores_dict = {title: value for title, value in zip(
            items_df[item_col], movie_to_list_avg_dissimilarity)}
        return dissimilarity_scores_dict
    except Exception as e:
        raise ValueError(
            f"An error occured while computing the diversity_score: {e}")



# diverify recommendations function
def diversifier(initial_recoms_dict, dissimilarity_scores_dict, theta_F=0.6, list_length=10):
    """_summary_

    Args:
        initial_recoms_dict (dict): the initial predicted ratings the user normalized between 0 and 1 
        dissimilarity_scores_dict (dict): the dissimialirty(diversity) of all items to the top list_length items in the initial_recoms_dict 
        theta_F (float, optional): weight for diversification. Defaults to 0.6.
        list_length (int, optional): length of the recommendations list. Defaults to 10.

    Returns:
        tuple(list,dict): list of the top list_length items after diversification, dict of the final diversified
        score of all movie titles sorted desc.
    """
    re_ranked_scores = {key: theta_F * dissimilarity_scores_dict[key] + (1-theta_F) * initial_recoms_dict[key] for key
                        in dissimilarity_scores_dict if key in initial_recoms_dict}
    re_ranked_scores_sorted = dict(
        sorted(re_ranked_scores.items(), key=lambda x: x[1], reverse=True))
    re_ranked_list = list(re_ranked_scores_sorted.keys())[:list_length]
    return re_ranked_list, re_ranked_scores_sorted



def diversity_main(user_id, user_data, combined_data, num_neighbors = 3):
    # get initial recommendations
    initial_recoms_series, initial_recomes_dict = movie_recommender(
        user_data, user_id, num_neighbors= 3, implicit_feedback_weight=0.5)
    
    data = data_reader(combined_data, common_columns=[
                       'title', 'description'])
    
    items_metadata = preprocessing(data, 'feature')
    cosine_matrix = similarity_matrix(items_metadata, 'feature')

    dissimilarity_scores_dict = diversity_score(user_id, user_data
    , cosine_matrix = cosine_matrix, items_df=items_metadata, item_col='title', n_neighbors=3, recoms_length=10)
    
    re_ranked_list, re_ranked_scores_sorted = diversifier(
        initial_recomes_dict, dissimilarity_scores_dict)
    
    
    return re_ranked_list, re_ranked_scores_sorted


def intralist_sim_score(items_df, cosine_matrix, recoms_list, item_col='title'):
    items_indices = items_df[items_df[item_col].isin(recoms_list)].index
    reranked_similarity_values = []
    for i in items_indices:
        for j in items_indices:
            if i != j:  # Skip comparison with itself
                reranked_similarity_values.append(cosine_matrix[i][j])
    score = sum(reranked_similarity_values) / len(reranked_similarity_values)
    return score
