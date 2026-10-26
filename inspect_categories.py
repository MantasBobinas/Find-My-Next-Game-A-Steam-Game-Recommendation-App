import csv
import pandas as pd

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    headings = next(csv.reader(file))

if "DiscountDLC count" in headings:
    position = headings.index("DiscountDLC count")
    headings[position:position + 1] = ["Discount", "DLC count"]

games = pd.read_csv(
    "data/games.csv",
    header=0,
    names=headings,
    usecols=["Name", "Categories"]
)

games = games.dropna(subset=["Name"]).copy()

print("Games checked:", len(games))
print("Missing categories:", games["Categories"].isna().sum())

category_lists = games["Categories"].fillna("").str.split(",")
categories = category_lists.explode().str.strip()
categories = categories[categories != ""]

print()
print("CATEGORY COUNTS:")
print(categories.value_counts().to_string())