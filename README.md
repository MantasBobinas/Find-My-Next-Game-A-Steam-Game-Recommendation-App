\# Find My Next Game



A Python application that recommends Steam games based on a written

description, genre, budget, and review preferences.



Built for the Artificial Intelligence and Data Analytics course.



\## Features



\- Search using phrases such as "relaxing farming crafting".

\- Filter by genre and maximum price in USD.

\- Set minimum positive-review percentage and total review count.

\- View up to five recommendations with Steam store links.



\## Tools and reasons for choosing them



\- Python: data preparation and recommendation logic.

\- pandas: loading, inspecting, and filtering tabular data.

\- Streamlit: building an interactive interface using Python.

\- scikit-learn: TF-IDF text representation and cosine similarity.

\- Git and GitHub: version history and project documentation.



TF-IDF provides an explainable starting point for text recommendations

without requiring a paid API or a large language model.



\## Dataset



Source: FronkonGames Steam Games Dataset

https://www.kaggle.com/datasets/fronkongames/steam-games-dataset



The local dataset contains 125,855 records.

After excluding one record with a missing name, 125,854 remain.



The CSV is not included in this repository. Download and extract the

dataset, create a folder named `data`, and place `games.csv` inside it.



Prices are in USD and reflect the dataset snapshot, not live Steam prices.



\## Data checks and cleaning



The downloaded CSV has 39 headings but 40 fields per record.

The heading `DiscountDLC count` combines two column names.

The loading code separates it into `Discount` and `DLC count`

without modifying the original CSV.



Checks on this dataset found:



\- All 125,855 records contain 40 fields.

\- One record has a missing name; it is excluded.

\- 8,423 records have missing genres.

\- No duplicate AppIDs.

\- No missing, non-numeric, or negative prices or review counts.



Games with missing genres remain eligible under "All genres".

Missing descriptions and tags are treated as empty text.

Games without reviews have an unavailable positive-review percentage.



\## How recommendations work



The app applies genre, budget, and review requirements.



With no search text, matching games are ranked by total review count.



With search text, the app:



1\. Combines each game's genres, tags, and description.

2\. Uses TF-IDF with English stop words and a maximum vocabulary

&#x20;  of 20,000 terms.

3\. Represents the user's request using the same vocabulary.

4\. Calculates cosine similarity between the request and each game.

5\. Excludes zero-similarity matches and ranks eligible games by

&#x20;  similarity, using review count to break ties.



At most five games are displayed. Similarity scores are not

probabilities that a user will enjoy a game.



\## Run locally on Windows



1\. Install Python and Git.

2\. Clone this repository and open Command Prompt in its folder.

3\. Run `py -m venv .venv`.

4\. Run `.venv\\Scripts\\activate.bat`.

5\. Run `python -m pip install -r requirements.txt`.

6\. Place the dataset at `data/games.csv`.

7\. Run `python -m streamlit run app.py`.



The current development environment uses Python 3.15.1.

Installed dependency versions are recorded in `requirements.txt`.



\## Project files



\- `app.py`: interactive recommendation application.

\- `check\_csv.py`: checks field counts throughout the CSV.

\- `inspect\_data.py`: checks missing values, duplicate IDs, and numbers.

\- `inspect\_text.py`: inspects descriptions and tags in a small sample.

\- `search\_demo.py`: tests text matching on the first 5,000 records.

\- `requirements.txt`: installed Python package versions.

\- `.gitignore`: excludes local data and the virtual environment.



\## Limitations and evaluation



TF-IDF matches vocabulary rather than fully understanding meaning.

It may miss synonyms or misunderstand requests such as "no combat".

Missing metadata can reduce recommendation quality.



Positive-review percentages are rounded to one decimal place before

filtering. A minimum review count does not guarantee reliable ratings.



Initial manual searches included farming simulation and civilization

building. These were informal checks; systematic evaluation is still

to be completed.



The dataset snapshot version and download date should be recorded

to make reproduction more precise.

