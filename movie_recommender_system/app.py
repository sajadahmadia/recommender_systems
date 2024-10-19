# import libraries
from math import comb
from typing import final
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from diversity import *
from coverage import *
from itertools import islice
import math
from coverage import *
import base64
import os
from streamlit_option_menu import option_menu

######################################################
# all necessary variables:
user_data_path = "data/ABC/userData.pkl"
combined_data_path = "data/ABC/combined.pkl"
user_id = "user_comedy lover_1"
num_neighbors = 3 
implicit_feedback_weights = 0.5

# final recommender function
def final_recommendations(user_id, user_data, combined_data, num_neighbors, diversity_coverage_balance, number_of_recommendations):
    # number of shows from each algorithm
    diversity_number = math.ceil(number_of_recommendations * diversity_coverage_balance)
    coverage_number = number_of_recommendations - diversity_number


    # diversity ranked shows
    diversity_names, diversity_names_scores_sorted = diversity_main(user_id, user_data, combined_data, num_neighbors)
        
    # list of diversity show names
    n_diversity_shows = diversity_names[:diversity_number]


    # get list of movies user has seen and likes
    liked_movies = get_liked_movies_from_user_data(user_data, user_id, movie_rating = 4, watched_threshold=85)
    
    # get remainder of recommended number of shows
    coverage_combined_similarity = combined_coverage_similarity(combined_data)
    # list of coverage show names
    nremainder_coverage_shows = recommend_with_individual_genre_selection(liked_movies, combined_data, coverage_combined_similarity, n_diversity_shows, coverage_number )[0]

    # combine the two arrays
    final_ranked_list = n_diversity_shows + nremainder_coverage_shows

    # lookup values of keys in dictionary and return a dictionary with the movie names and scores 
    final_ranked_dict = {}
    for item in final_ranked_list:
        final_ranked_dict[item] = diversity_names_scores_sorted[item]

    # return distionary of recommended movies and dictionary of all movies with score
    return final_ranked_dict, diversity_names_scores_sorted


# template functions
def display_user_profile(user_id):
    user_name = user_profiles[user_id]['name']
    user_preferences = user_profiles[user_id]["preferences"]
    
    # default user image
    user_image_path = "images/default_person.png"  
    
    # get user image path
    user_image_path = user_profiles[user_id]["image"]

    # Check if the image exists before attempting to display it
    try:
        with open(user_image_path, "rb") as file:
            user_image = file.read()
            st.image(user_image, width=100)
    except FileNotFoundError:
        st.warning("User image not found.")
    
    # Display the rest of the user profile information
    st.write(f"**Name:** {user_name}")
    st.write(f"**Preferences:** {user_preferences}")

          

def get_movie_image_url(movie_title, combined_data):
    # Modify 'title' to the actual column name
    movie_row = combined_data[combined_data['title'] == movie_title]  
    if not movie_row.empty:
        return movie_row.iloc[0]['image']
    else:
        return None


# Page configuration
st.set_page_config(
    page_title='MyMovie',
    layout='wide',
    page_icon="🎬",
    initial_sidebar_state="expanded"
)


## User profiles definition
user_profiles = {
    "user_comedy_lover_1": {
        "name": "Anna Soloman",
        "age": 24,
        "image": "images/anna.png",
        "characteristics": "Loves comedy movies, nature, and travel",
        "preferences": {"genre": "Comedy"},
        "coverage_weight": 0.5,
        "diversity_weight": 0.5
    },
    "user_young_families_1": {
        "name": "John Doe",
        "age": 54,
        "image": "images/john_doe.png",
        "characteristics": "Enjoys family-friendly films, cooking, and board games",
        "preferences": {"genre": "Family"},
        "coverage_weight": 0.5,
        "diversity_weight": 0.5
    },
    "user_drama_hater_3": {
        "name": "Elisabeth Handock",
        "age": 34,
        "image": "images/elizabeth.png",
        "characteristics": "Avoids drama movies",
        "preferences": {"exclude_genre": "Drama"},
        "coverage_weight": 0.5,
        "diversity_weight": 0.5
    },
    "user_movie_buff_1": {
        "name": "Movie Buff Mike",
        "age": 29,
        "image": "images/movie_buff_mike.png",
        "characteristics": "Watches all genres, loves film festivals",
        "preferences": {"genre": "All"},
        "coverage_weight": 0.5,
        "diversity_weight": 0.5
    },
}



def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


######################################################
# webpage templates
def home_page():
    st.title("Home Page")
    st.write("Welcome to the MyMovie app!")
    
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
        

def personas_page():
    st.title("Personas")
    
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    
    # Create two sets of columns for a 2x2 grid
    cols = st.columns(2)  # Creates 2 columns for the top row
    user_profile_keys = list(user_profiles.keys())

    # Assuming you have exactly 4 profiles
    for index in range(0, 4, 2):  # step by 2 since we display 2 profiles per row
        # First row
        with cols[0]:
            display_persona(user_profiles[user_profile_keys[index]])
        with cols[1]:
            display_persona(user_profiles[user_profile_keys[index + 1]])

        if index == 0:  # Only create a new row after the first set of columns
            cols = st.columns(2)  # Creates another 2 columns for the bottom row


def display_persona(user_profile):
    st.image(user_profile["image"], width=150, caption=user_profile["name"])
    st.write(f"**Name:** {user_profile['name']}")
    st.write(f"**Age:** {user_profile['age']}")
    st.write("**Characteristics:**")
    if isinstance(user_profile['characteristics'], list):  # Check if characteristics is a list
        for characteristic in user_profile['characteristics']:
            st.write(f"- {characteristic}")
    else:  # If it's a string instead, just print the string
        st.write(user_profile['characteristics'])


def recommendations_page():
    st.title("Recommendations")
    st.write("Movies recommended based on your preferences will appear here.")
    
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    
    
    
    # Sidebar section for user profile selection
    st.sidebar.subheader('User Profile Selection')
    selected_user_persona = st.sidebar.selectbox('Select user persona', options=list(user_profiles.keys()), index=0)
    user_profile = user_profiles[selected_user_persona]
    st.sidebar.image(user_profile["image"], width=150)
    st.sidebar.caption(user_profile["name"])
    st.sidebar.write(f"Characteristics: {user_profile['characteristics']}")
    
    # Sidebar section for combined slider
    st.sidebar.subheader('Adjust Balance')
     
     # Create a single slider for adjusting the balance between Diversity and Coverage
     # The value will represent the weight of Diversity, with the remainder being the weight for Coverage.
    diversity_weight = st.sidebar.slider('Diversity ↔ Coverage Balance:', 0.0, 1.0, 0.5, format="%.2f")
     
     # Calculate the coverage weight based on the diversity weight
    coverage_weight = 1.0 - diversity_weight
     
     # Display the calculated weights for clarity
    st.sidebar.write(f"Diversity Weight: {diversity_weight:.2f}")
    st.sidebar.write(f"Coverage Weight: {coverage_weight:.2f}")
    number_of_recommendations = st.sidebar.slider('Number of Recommendations:', 1, 20, 10)

    
    try:
        # load combined data    
        combined_data = pd.read_pickle(combined_data_path)
        # load user data
        user_data = pd.read_pickle(user_data_path)

        # get n sorted top ranked shows for user
        recommended_shows, diversity_names_scores_sorted = final_recommendations(user_id, user_data, combined_data, num_neighbors, diversity_weight, number_of_recommendations)
        
        st.write("### Recommended Shows")
        container = st.container()
        with container:
            number_of_columns = 5
            columns = st.columns(number_of_columns)
            column_index = 0
            for title in recommended_shows:
                # Get the score, ensuring it's either a float or "N/A"
                score = diversity_names_scores_sorted.get(title, "N/A")
                
                image_url = get_movie_image_url(title, combined_data)  # Get the image
                with columns[column_index]:
                    st.image(image_url, width=200)
                    
                    # Only format the score if it's not "N/A"
                    if score != "N/A":
                        st.caption(f"{title} - Score: {score:.2f}")
                    else:
                        st.caption(f"{title} - Score: N/A")
        
                column_index = (column_index + 1) % number_of_columns
                if column_index == 0:
                    columns = st.columns(number_of_columns)
                

        st.write("### Catalog Coverage")
        # Convert counts to proportions
        total_recommendations = sum(recommendation_counts.values()) 
        proportions = np.array(list(recommendation_counts.values())) / total_recommendations

        coverage = catalog_coverage(combined_data)
        gini_index = calculate_gini_index(proportions)
        shannon_entropy = calculate_shannon_entropy(recommendation_counts)

        st.write(f"Catalog Coverage: {coverage:.2f}")
        st.write(f"Gini Index: {gini_index:.2f}")
        st.write(f"Shannon Entropy: {shannon_entropy:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    

def explanations_page():
    st.title("Explanations")
    st.write("""
            Imagine you're browsing through a vast library of videos, each unique in its story, characters, and genre. Our recommendation system is like a friendly librarian who knows your tastes and preferences, guiding you to content you'll likely enjoy. Here's how it works in simple terms:

            We start by analyzing all the videos using two smart algorithms—one focuses on making sure you see a wide variety of content (diversity), and the other ensures you're not missing out on anything (coverage). These algorithms look at the descriptions, tags, and even how other viewers similar to you have rated these videos to understand what you might like.

            First, our system notes down which videos you've watched most of and liked (watched over 75% and rated 4 or higher). Then, it compares these to the rest of our video catalog, looking for similar ones based on the content and categories. But here's the twist: we also sprinkle in some videos from genres you might not watch often, ensuring you discover new favorites and not just the popular picks.

            For diversity, we find other users who have similar tastes and check out what they loved. This way, you get recommendations that are a hit among those with preferences like yours but might include hidden gems you haven't stumbled upon yet.

            Finally, we blend these insights together, balancing your love for what's familiar and the thrill of discovering something new. Through our user-friendly interface, you can even tweak how much variety you want in your recommendations, making sure it's just right for you.
            """)
    
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
        
    
    st.sidebar.markdown("---")


# Define the option menu and handle page navigation
selected = option_menu(
    menu_title=None,  # No title for the menu
    options=["Home", "Personas", "Recommendations", "Explanations"],  # Names of the pages
    icons=["house", "users", "film", "book"],  # Corresponding icons
    menu_icon="cast",  # Menu icon
    default_index=0,  # Default page to show
    orientation="horizontal",  # Horizontal menu
)

# Call the corresponding function based on the selected option
if selected == "Home":
    home_page()
elif selected == "Personas":
    personas_page()
elif selected == "Recommendations":
    recommendations_page()
elif selected == "Explanations":
    explanations_page()



    