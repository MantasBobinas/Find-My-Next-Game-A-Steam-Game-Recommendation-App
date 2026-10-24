import csv

with open("data/games.csv", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    header = next(reader)

    print("Number of headings:", len(header))
    print("First 8 headings:", header[:8])

    for row_number in range(1, 4):
        row = next(reader)
        print()
        print("Row:", row_number)
        print("Number of values:", len(row))
        print("First 8 values:", row[:8])