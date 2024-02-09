# MovieRecommendationSystem

## Here's a summary of the key steps involved:

* Data Loading: The code loads two CSV files: one containing movie information and another containing credits information.
* Data Merging and Cleaning: The two datasets are merged based on the "title" field. Missing values in the "overview" column are dropped, and duplicates are removed.
* Data Preprocessing:
     Genres, keywords, cast, and crew information are extracted from their respective string formats into lists.

     The top 3 cast members are selected for each movie.

     Directors are extracted from the crew information.

     Unnecessary spaces in cast and crew names are removed.

     All text data (overview, genres, keywords, cast, and crew) are converted to lowercase and stemmed (reduced to their root form) to improve similarity 
     comparison.

* Feature Engineering:
     A new "tag" column is created by combining all the preprocessed text data into a single string.

     A CountVectorizer is used to convert the text data in the "tag" column into numerical vectors.

* Similarity Calculation:
     Cosine similarity is calculated between all movie vectors to measure their similarity.

* Recommendation System:
     A function is defined to recommend movies similar to a given movie title.

     The function retrieves the similarity vector for the specified movie and finds the movies with the highest similarity scores.

     The titles of the most similar movies are then returned as recommendations.

Overall, this code demonstrates a content-based recommendation system for movies, utilizing various data preprocessing and text analysis techniques to achieve relevant recommendations.
