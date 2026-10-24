import csv
import pandas as pd

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    headings = next(csv.reader(file))

position = headings.index("DiscountDLC count")
headings[position:position + 1] = ["Discount", "DLC count"]

games = pd.read_csv(
    "data/games.csv",
    header=0,
    names=headings,
    nrows=5
)

print("Corrected heading count:", len(headings))

selected_columns = [
    "AppID", "Name", "Release date", "Price",
    "Discount", "DLC count", "Genres"
]

print(games[selected_columns].to_string(index=False))