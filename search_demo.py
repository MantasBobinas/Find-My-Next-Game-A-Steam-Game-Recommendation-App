import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    headings = next(csv.reader(file))

position = headings.index("DiscountDLC count")
headings[position:position + 1] = ["Discount", "DLC count"]

games = pd.read_csv(
    "data/games.csv",
    header=0,
    names=headings,
    usecols=["Name", "Genres", "Tags", "About the game"],
    nrows=5000
)

games = games.dropna(subset=["Name"]).copy()

search_text = (
    games["Genres"].fillna("") + " "
    + games["Tags"].fillna("") + " "
    + games["About the game"].fillna("")
)

print("Preparing the search...")

vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
game_vectors = vectorizer.fit_transform(search_text)

query = input("Describe the game you want: ").strip()
query_vector = vectorizer.transform([query])

scores = cosine_similarity(query_vector, game_vectors).ravel()
games["Similarity"] = scores

matches = games[games["Similarity"] > 0]
results = matches.sort_values(by="Similarity", ascending=False).head(5)

if results.empty:
    print("No matching words found. Try different search terms.")
else:
    print(results[["Name", "Genres", "Similarity"]].to_string(index=False))