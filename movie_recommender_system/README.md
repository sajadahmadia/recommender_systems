# Movie Recommender System

* Dataset: the recommender system is a dataset of 13885 articles aggregated from various video-based content.
* Algorithm: The system is made up of two separate algorithms: one for operationalizing diversity and one for coverage as public values. These algorithms are later combined to give the final recommendations within a transparent interface, thereby operationalizing transparency.
  - Coverage: to improve the domain of the recommended items
  - Diversity: to improve the range of the content that users encounter through recommendations from the system

## Coverage Algorithm:
**Data Preprocessing**  
Duplicates were removed, tags and descriptions standardized, and stop-words were removed.

**Feature Extraction**  
A combination of text-based and categorical features has been implemented:
- **TF-IDF Vectorization**: For textual data such as descriptions and tags, TF-IDF vectorization is used to transform text into a feature matrix.
- **One-Hot Encoding**: Categorical data (ratings, e.g., “PG”) are encoded using one-hot encoding, facilitating similarity measurement between items.

**Similarity Measurement**  
The system employs cosine similarity for text-based features and Jaccard similarity for categorical features. These similarities are normalized and combined to get the final aggregated similarity scores.

**Operationalization**  
The recommendation algorithm takes movies the user likely enjoys, defined as those watched over 75% and rated four or higher. It calculates a mean similarity score for these against the catalogue, outputting a ranked list of movies by similarity.

The system then employs randomized genre selection to ensure diverse genre exposure, enhancing catalogue coverage while still recommending the most similar content of the selected (random) genre. It also modifies recommendation scores based on past recommendation frequency, promoting lesser-known content to avoid popularity bias and achieve broad catalog representation.


## Diversity Algorithm
**Step 1**

The system initiates by creating a matrix mapping movie IDs to user IDs. It then applies a k-nearest neighbors (kNN) algorithm to identify three users most similar to the target user. Leveraging these findings, the algorithm determines the top 10 movies rated by these similar users.


**Step 2**

- **Preprocessing**  
The feature content (title and description) per row was combined into a single string and lemmatized.

- **Feature Extraction**  
TF-IDF vectorization was used to transform the single string into a feature matrix.

- **Similarity Measurement**  
A cosine similarity matrix is computed from the TF-IDF matrix to quantify the similarity between each pair of items in the dataset.

- **Recommendation Generation**  
Given the output list of 10 movies from Step 1, the system then generates an ordered list of recommendations based on the cosine similarity scores between these 10 movies and all other movies.

- **Diversity Enhancement**  
With the same output list of 10 movies from Step 1, the system refines recommendations by calculating each item’s dissimilarity to others, applying a diversity factor (determined by the user) to balance similarity and diversity (dissimilarity). Items are then re-ranked for the final recommendation dictionary based on their user-defined relevance and diversity contribution.

## Combining the Algorithms
The final recommendations combine outputs from one dictionary of movies with similarity scores and a coverage function call, tailored to the user’s preference for diversity versus coverage. If the user prefers diversity, recommendations lean more towards the diverse content from the dictionary; if coverage is preferred, the system generates more recommendations from the coverage function.

The final output is a set of N recommendations which incorporates diversity and coverage as public values. These recommendations are then displayed to the user in a transparent way, which is discussed in the following section.

## Interface Desging: Improving Transparency
The interface enables users to choose personas that reflect their tastes, affecting the recommendations they receive. Users can fine-tune their experience using a slider to balance the impact of coverage and diversity algorithms on their recommendations. Additionally, the interface shows similarity scores for recommended movies, providing insights into their alignment with the user’s selected persona. A section detailing the recommender system’s mechanics offers a transparent view into how and why recommendations are made, enhancing user understanding and involvement.

## How to Implement the Algorithm?
1. Install streamlit library using `pip install streamlit`
2. Clone the current repository on your machine
3. Go to the repository's directory
4. Run streamlit using `streamlit run app.py`

* Notice: Please preserve the current hierarchy of the sub-directories and make no changes to them.

**Reference:**
The system is the result of teamwork as part of the Personalization for Public Media course provided at Utrecht University. I was responsible for writing the `diversity algorithm`. The rest of the team members were:
* Fin Zandbergen and Madio Seck: responsible for writing the coverage algorithm
* David Sijbesma: responsible for creating the UI
* Madio Seck: responsible for combining the algorithms
