# prompt: create a user interface for movie recommendation system  with pictures in it

import difflib
from io import BytesIO
from math import comb

import ipywidgets as widgets
import numpy as np
import pandas as pd
import requests
from IPython.display import clear_output, display
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset
movies_data = pd.read_csv('/content/movies.csv')

# Select relevant features and handle missing values
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')

# Combine features and create TF-IDF vectors
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

def get_movie_poster(movie_title):
  try:
    # Search for the movie on TMDB
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key=YOUR_API_KEY&query={movie_title}"
    response = requests.get(search_url)
    data = response.json()
    if data['results']:
      poster_path = data['results'][0]['poster_path']
      poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
      return poster_url
    else:
      return None
  except:
    return None

def recommend_movies(movie_name):
  list_of_all_titles = movies_data['title'].tolist()
  find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
  if find_close_match:
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Display recommendations with posters
    output.clear_output()
    with output:
      display(widgets.HTML(value=f"<h3>Movies suggested for you based on '{close_match}':</h3>"))
      for i, movie in enumerate(sorted_similar_movies[1:10]):
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        poster_url = get_movie_poster(title_from_index)

        if poster_url:
          image = Image.open(BytesIO(requests.get(poster_url).content))
          display(widgets.HBox([widgets.Image(value=image._repr_png_(), width=100, height=150), widgets.HTML(value=f"<b>{i+1}. {title_from_index}</b>")]))
        else:
          display(widgets.HTML(value=f"<b>{i+1}. {title_from_index}</b>"))
  else:
    with output:
      display(widgets.HTML(value="Movie not found in database."))

# Create input widget and output area
movie_input = widgets.Text(placeholder='Enter your favourite movie name')
recommend_button = widgets.Button(description='Recommend')
output = widgets.Output()

# Define button click event
def on_button_clicked(b):
  recommend_movies(movie_input.value)

recommend_button.on_click(on_button_clicked)

# Display the UI
display(widgets.VBox([movie_input, recommend_button, output]))


