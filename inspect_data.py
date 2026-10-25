import csv
import pandas as pd

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    headings = next(csv.reader(file))

position = headings.index("DiscountDLC count")
headings[position:position + 1] = ["Discount", "DLC count"]

columns = ["AppID", "Name", "Price", "Genres", "Positive", "Negative"]

games = pd.read_csv(
    "data/games.csv",
    header=0,
    names=headings,
    usecols=columns
)

print("MISSING VALUES:")
print(games.isna().sum())

print()
print("Duplicate game IDs:", games["AppID"].duplicated().sum())

print()
print("NUMERIC CHECKS:")

for column in ["Price", "Positive", "Negative"]:
    numbers = pd.to_numeric(games[column], errors="coerce")

    invalid = games[column].notna() & numbers.isna()

    print(column)
    print("  Non-numeric values:", invalid.sum())
    print("  Negative values:", (numbers < 0).sum())