# Find My Next Game

A Steam game recommendation app built for the **Artificial Intelligence and Data Analytics** course.

Describe the game you want, choose your preferences, and discover matching games from a dataset of over **125,000 Steam titles**.

Built with **Python**, **pandas**, **Streamlit**, and **scikit-learn**.

## What It Does

- Searches game descriptions, tags, and genres using TF-IDF and cosine similarity.
- Filters games by genre and maximum price in USD.
- Supports minimum positive-review percentages and review counts.
- Displays up to five recommendations with review information.
- Provides clickable links to each game's Steam store page.
- Caches loaded data and the search index for reuse between interactions.

## Project Structure

- `app.py`: Streamlit interface, data preparation, filters, and recommendations.
- `requirements.txt`: installed Python package versions.
- `.gitignore`: excludes local datasets and the virtual environment.
- `data/games.csv`: downloaded dataset, stored locally and excluded from Git.

## Setup (Windows Command Prompt)

### 1. Download the project

```bat
git clone https://github.com/MantasBobinas/Find-My-Next-Game-A-Steam-Game-Recommendation-App.git
cd Find-My-Next-Game-A-Steam-Game-Recommendation-App
```

### 2. Create and activate a virtual environment

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

The project was developed using **Python 3.15.1**.

### 3. Install dependencies

```bat
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Download the [FronkonGames Steam Games Dataset](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) and extract the archive.

Create the local data folder:

```bat
mkdir data
```

Copy `games.csv` into it. The expected location is `data/games.csv`.

The dataset is not included in this repository.

## Run

With the virtual environment active, launch the app:

```bat
python -m streamlit run app.py
```

Open the Local URL printed in the terminal if the browser does not open automatically.

Keep the terminal open while using the app. Press **Ctrl + C** to stop it.

## How to Use

1. Optionally describe a game, such as `relaxing farming crafting`.
2. Choose a genre or leave **All genres** selected.
3. Set a maximum price in USD.
4. Choose minimum positive-review percentage and review count.
5. Review the recommendations and select **Open on Steam**.

Leave the description blank to discover matching games ranked by review count.

## Recommendation Method

### With a written request

The app combines each game's genres, tags, and description into searchable text.

**TF-IDF** represents that text using weighted words, with English stop words removed and a vocabulary limited to 20,000 terms. **Cosine similarity** compares the request with each game.

Games must satisfy the selected filters and have a non-zero text similarity. Results are ranked by similarity, with total review count used to break ties.

### Without a written request

Games are filtered by the selected preferences and ranked by total review count.

### Review information

- **Total reviews:** positive reviews plus negative reviews.
- **Positive-review percentage:** positive reviews divided by total reviews, multiplied by 100.
- Games without reviews have an unavailable positive-review percentage.
- Percentage filtering uses values rounded to one decimal place.

## Data Quality and Cleaning

The inspected dataset contains **125,855 records**.

The original CSV has a joined heading, `DiscountDLC count`. The loading code separates it into `Discount` and `DLC count`, producing 40 headings that match the 40 fields in every record.

Checks found:

- **1 missing game name:** excluded, leaving **125,854 games**.
- **8,423 missing genres:** retained under **All genres**.
- **0 duplicate AppIDs**.
- **0 missing prices or review counts**.
- **0 non-numeric or negative prices or review counts**.

Missing text fields are treated as empty text. The original CSV is not modified.

Run the data checks with:

```bat
python check_csv.py
python inspect_data.py
```

## Why These Tools

- **pandas:** practical tools for inspecting and filtering tabular data.
- **Streamlit:** an interactive browser interface built directly in Python.
- **scikit-learn:** established tools for an explainable text-matching baseline.
- **Git and GitHub:** version history, documentation, and reproducible setup instructions.

The recommendation method runs locally without a paid AI API.

## Limitations

- TF-IDF matches vocabulary rather than fully understanding meaning.
- Synonyms and negative requests such as “no combat” may be handled poorly.
- Similarity scores are not probabilities of enjoying a game.
- Missing descriptions or tags can reduce recommendation quality.
- Prices reflect the dataset snapshot and may differ from current Steam prices.
- Review volume measures attention; it does not guarantee quality.
- The dataset version and download date still need to be recorded for more precise reproduction.

## Evaluation Status

Initial manual checks covered farming simulation and civilization-building searches, preference filters, and a Steam store link.

These checks confirm initial functionality. A systematic relevance evaluation and comparison with the review-count baseline are still planned.
