import csv
import pandas as pd
import streamlit as st

st.title("Find My Next Game")
st.write("Explore a sample of the Steam dataset.")

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    headings = next(csv.reader(file))

position = headings.index("DiscountDLC count")
headings[position:position + 1] = ["Discount", "DLC count"]

games = pd.read_csv(
    "data/games.csv",
    header=0,
    names=headings,
    nrows=100
)

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

recommendations = filtered_games.sort_values(
    by="Total reviews",
    ascending=False
).head(5)

st.write("Games matching your filters:", len(filtered_games))

if recommendations.empty:
    st.info("No matches found. Try another genre or a higher budget.")
else:
    st.subheader("Your game recommendations")

    display_columns = ["Name", "Price", "Genres", "Total reviews"]

    st.dataframe(
        recommendations[display_columns],
        hide_index=True
    )

st.caption(
    "Searching the first 100 records. Ranked by total review count. "
    "Prices are in USD from the dataset snapshot and may differ from current Steam prices."
)