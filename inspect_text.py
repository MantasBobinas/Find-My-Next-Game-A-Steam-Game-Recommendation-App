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
    usecols=["Name", "About the game", "Tags"],
    nrows=20
)

print("Missing values in this 20-record sample:")
print(games.isna().sum())

examples = games.dropna(subset=["Name", "About the game", "Tags"]).head(3)

for _, game in examples.iterrows():
    print()
    print("NAME:", game["Name"])
    print("DESCRIPTION:", str(game["About the game"])[:600])
    print("TAGS:", game["Tags"])