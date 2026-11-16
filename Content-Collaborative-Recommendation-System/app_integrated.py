import os
import pickle

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# 1. Page configuration and TMDB API setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.title("Movie Recommender System")

# Set TMDB_API_KEY in the environment before starting Streamlit.
# Do not hardcode or publish your API key in this file.
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()

if not TMDB_API_KEY:
    st.warning(
        "TMDB poster images require an API key. Set the TMDB_API_KEY "
        "environment variable before starting Streamlit."
    )


# ---------------------------------------------------------
# 2. Load the existing Content-Based model and new
#    Collaborative Filtering model
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    with open("/content/movies_list.pkl", "rb") as file:
        movies_df = pickle.load(file)

    with open("/content/similarity.pkl", "rb") as file:
        content_similarity = pickle.load(file)

    with open("/content/collaborative_model.pkl", "rb") as file:
        collaborative_objects = pickle.load(file)

    return movies_df, content_similarity, collaborative_objects


try:
    movies, similarity, collaborative_objects = load_models()
except FileNotFoundError as error:
    st.error(
        f"Required model file not found: {error.filename}. "
        "Check that movies_list.pkl, similarity.pkl and "
        "collaborative_model.pkl are present in /content."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the recommendation models: {error}")
    st.stop()


movies_list = movies["title"].dropna().astype(str).tolist()

cf_model = collaborative_objects["model"]
item_user_matrix = collaborative_objects["matrix"]
item_user_sparse = collaborative_objects["sparse_matrix"]
movie_lookup = collaborative_objects["movie_lookup"]


# ---------------------------------------------------------
# 3. Fetch movie posters from TMDB
# ---------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")

        if not poster_path:
            return None

        return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except (requests.RequestException, ValueError, TypeError):
        return None


# ---------------------------------------------------------
# 4. Existing Content-Based Filtering function
# ---------------------------------------------------------
def recommend_content_based(movie_title, n=5):
    matches = movies.index[movies["title"] == movie_title]
    if len(matches) == 0:
        return []

    movie_index = matches[0]
    distances = sorted(
        enumerate(similarity[movie_index]),
        key=lambda item: item[1],
        reverse=True
    )

    recommendations = []
    for index, score in distances[1:n + 1]:
        movie_row = movies.iloc[index]
        recommendations.append({
            "title": str(movie_row["title"]),
            "tmdb_id": int(movie_row["id"]),
            "similarity": float(score)
        })

    return recommendations


# ---------------------------------------------------------
# 5. New Item-Based Collaborative Filtering function
# ---------------------------------------------------------
def recommend_collaborative(movie_title, n=5):
    title_matches = movie_lookup[
        movie_lookup["title"].astype(str).str.casefold()
        == str(movie_title).casefold()
    ]

    if title_matches.empty:
        return []

    # movie_lookup is indexed by TMDB ID.
    movie_id = int(title_matches.index[0])
    if movie_id not in item_user_matrix.index:
        return []

    movie_index = item_user_matrix.index.get_loc(movie_id)
    number_to_find = min(n + 1, item_user_matrix.shape[0])

    distances, indices = cf_model.kneighbors(
        item_user_sparse[movie_index],
        n_neighbors=number_to_find
    )

    recommendations = []
    for neighbor_index, distance in zip(
        indices.flatten(), distances.flatten()
    ):
        recommended_id = int(item_user_matrix.index[neighbor_index])
        if recommended_id == movie_id:
            continue
        if recommended_id not in movie_lookup.index:
            continue

        recommended_title = movie_lookup.loc[recommended_id, "title"]
        if isinstance(recommended_title, pd.Series):
            recommended_title = recommended_title.iloc[0]

        recommendations.append({
            "title": str(recommended_title),
            "tmdb_id": recommended_id,
            "similarity": float(1 - distance)
        })

        if len(recommendations) >= n:
            break

    return recommendations


# ---------------------------------------------------------
# 6. Streamlit user interface
# ---------------------------------------------------------
st.subheader("Choose a recommendation technique")

recommendation_method = st.radio(
    "Recommendation method",
    ["Content-Based Filtering", "Collaborative Filtering"],
    horizontal=True
)

selected_movie = st.selectbox("Select movie from dropdown", movies_list)

if st.button("Show Recommend", type="primary"):
    with st.spinner("Generating movie recommendations..."):
        if recommendation_method == "Content-Based Filtering":
            recommendations = recommend_content_based(selected_movie)
        else:
            recommendations = recommend_collaborative(selected_movie)

    if not recommendations:
        if recommendation_method == "Collaborative Filtering":
            st.warning(
                "This movie is not available in the Collaborative Filtering "
                "rating dataset. Try another movie, such as The Godfather "
                "or Iron Man."
            )
        else:
            st.warning("No recommendations were found for this movie.")
    else:
        st.subheader(f"Recommended movies for: {selected_movie}")
        columns = st.columns(len(recommendations))

        for column, recommendation in zip(columns, recommendations):
            with column:
                st.caption(recommendation["title"])
                poster_url = fetch_poster(recommendation["tmdb_id"])

                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.info("Poster unavailable")

                st.caption(f"Similarity: {recommendation['similarity']:.3f}")


# The optional custom JavaScript carousel from the original app is omitted
# because its component JavaScript was producing a browser module-loading error.
# The original Content-Based model and poster display are retained.
