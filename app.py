from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
import streamlit as st

st.title("Find My Next Game")
st.write("Find Steam games that match your genre and budget.")

@st.cache_data
def load_games():
    with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
        headings = next(csv.reader(file))

    position = headings.index("DiscountDLC count")
    headings[position:position + 1] = ["Discount", "DLC count"]




    needed_columns = [
        "AppID", "Name", "Price", "Genres", "Positive", "Negative",
        "About the game", "Tags"
    ]

    data = pd.read_csv(
        "data/games.csv",
        header=0,
        names=headings,
        usecols=needed_columns
    )

    data = data.dropna(subset=["Name"]).copy()
    data = data.reset_index(drop=True)

    data["Search text"] = (
        data["Genres"].fillna("") + " "
        + data["Tags"].fillna("") + " "
        + data["About the game"].fillna("")
    )

    return data

@st.cache_resource
def build_search_index(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )
    game_vectors = vectorizer.fit_transform(texts)
    return vectorizer, game_vectors
games = load_games()
st.write("Games available after cleaning:", len(games))
query = st.text_input(
    "Describe the game you want",
    placeholder="For example: relaxing farming crafting"
).strip()

genre_lists = games["Genres"].fillna("").str.split(",")

available_genres = set()

for genres in genre_lists:
    for genre in genres:
        if genre:
            available_genres.add(genre)

genre_options = ["All genres"] + sorted(available_genres)

selected_genre = st.selectbox("Choose a genre", genre_options)

filtered_games = games

if selected_genre != "All genres":
    matches = genre_lists.apply(lambda genres: selected_genre in genres)
    filtered_games = games[matches]

max_price = st.number_input(
    "Maximum price (USD)",
    min_value=0.0,
    value=20.0,
    step=1.0
)

filtered_games = filtered_games[
    filtered_games["Price"] <= max_price
]

filtered_games = filtered_games.copy()

filtered_games["Total reviews"] = (
    filtered_games["Positive"] + filtered_games["Negative"]
)
review_totals = filtered_games["Total reviews"].replace(0, float("nan"))

filtered_games["Positive reviews (%)"] = (
    filtered_games["Positive"] / review_totals * 100
).round(1)
min_positive = st.number_input(
    "Minimum positive reviews (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)

if min_positive > 0:
    filtered_games = filtered_games[
        filtered_games["Positive reviews (%)"] >= min_positive
    ]
min_reviews = st.number_input(
    "Minimum number of reviews",
    min_value=0,
    value=100,
    step=100
)

filtered_games = filtered_games[
    filtered_games["Total reviews"] >= min_reviews
]
if query:
    vectorizer, game_vectors = build_search_index(games["Search text"])
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, game_vectors).ravel()

    filtered_games = filtered_games.copy()
    filtered_games["Text similarity"] = scores[filtered_games.index]
    filtered_games = filtered_games[
        filtered_games["Text similarity"] > 0
    ]

    recommendations = filtered_games.sort_values(
        by=["Text similarity", "Total reviews"],
        ascending=[False, False]
    ).head(5)
else:
    recommendations = filtered_games.sort_values(
        by="Total reviews",
        ascending=False
    ).head(5)

st.write("Games matching your filters:", len(filtered_games))

if recommendations.empty:
    st.caption(
    "With search text: ranked by word similarity, then review count. "
    "Without search text: ranked by review count. "
    "Prices are in USD from the dataset snapshot and may differ from current Steam prices."
)
else:
    st.subheader("Your game recommendations")

    recommendations = recommendations.copy()

    recommendations["Steam link"] = (
        "https://store.steampowered.com/app/"
        + recommendations["AppID"].astype(int).astype(str)
        + "/"
    )

    display_columns = [
        "Name", "Price", "Genres", "Total reviews",
        "Positive reviews (%)", "Steam link"
    ]

    st.dataframe(
        recommendations[display_columns],
        hide_index=True,
        column_config={
            "Steam link": st.column_config.LinkColumn(
                "Steam store",
                display_text="Open on Steam"
            )
        }
    )

st.caption(
    "With search text: ranked by word similarity, then review count. "
    "Without search text: ranked by review count. "
    "Prices are in USD from the dataset snapshot and may differ from current Steam prices."
)