import csv

csv.field_size_limit(10_000_000)

total_games = 0
mismatched_rows = 0

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    headings = next(reader)

    position = headings.index("DiscountDLC count")
    headings[position:position + 1] = ["Discount", "DLC count"]

    for row in reader:
        total_games += 1

        if len(row) != len(headings):
            mismatched_rows += 1

            if mismatched_rows <= 5:
                print("Unexpected field count in game record:", total_games)

print("Expected fields per game:", len(headings))
print("Total game records:", total_games)
print("Records with unexpected field counts:", mismatched_rows)