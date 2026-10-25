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
        "AppID", "Name", "Price", "Genres", "Positive", "Negative"
    ]

    data = pd.read_csv(
        "data/games.csv",
        header=0,
        names=headings,
        usecols=needed_columns
    )

    data = data.dropna(subset=["Name"]).copy()

    return data


games = load_games()
st.write("Games available after cleaning:", len(games))

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
recommendations = filtered_games.sort_values(
    by="Total reviews",
    ascending=False
).head(5)

st.write("Games matching your filters:", len(filtered_games))

if recommendations.empty:
    st.info("No matches found. Try another genre or a higher budget.")
else:
    st.subheader("Your game recommendations")

    display_columns = [
        "Name", "Price", "Genres", "Total reviews", "Positive reviews (%)"
    ]

    st.dataframe(
        recommendations[display_columns],
        hide_index=True
    )

st.caption(
    "Ranked by total review count. Prices are in USD from the "
    "dataset snapshot and may differ from current Steam prices."
)