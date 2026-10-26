"""Find My Next Game — Streamlit interface and recommendation baseline.
Run from your project folder: python -m streamlit run app.py
"""
import csv
import html
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Find My Next Game", page_icon="🎮", layout="wide")

# CSS changes presentation only. Dataset text is escaped before entering HTML.
STYLE = """
<style>
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #1b2235 !important;
    color: #edf0fa !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] > div * {
    color: #edf0fa !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #a8b2cc !important;
    opacity: 1;
}
:root {color-scheme: dark;}
.stApp {background: radial-gradient(ellipse at 85% 0%, #272044 0%, transparent 45%), #0c101b; color:#edf0fa;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#121725; border-right:1px solid #272d40;}
[data-testid="stSidebar"] * {color:#e5e8f5;}
[data-testid="stMainBlockContainer"] {max-width:1260px; padding-top:3rem; padding-bottom:3rem;}
h1,h2,h3,p,label {color:inherit;}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {background:#1b2235;color:#f1f3ff;}
[data-testid="stNumberInput"] button {background:#242c43;color:#f1f3ff;}
[data-baseweb="select"] > div {background:#1b2235;color:#f1f3ff;border-color:#39415b;}
[data-baseweb="popover"] li {color:#182033;}
[data-testid="stWidgetLabel"] p {color:#d8def0;}
[data-testid="stCaptionContainer"] {color:#a8b2cc;}
[data-testid="stTextInput"] > div {border-radius:14px;border-color:#685594;}
[data-testid="stButton"] button {background:#25203b;color:#e5d9ff;border:1px solid #51416e;border-radius:999px;}
[data-testid="stButton"] button:hover {border-color:#c2a0ff;color:white;background:#382953;}
[data-testid="stExpander"] {background:#121725;border-color:#30374d;color:#dae0ef;}
.eyebrow {font-size:.73rem;letter-spacing:.18em;font-weight:700;color:#bba0ff;text-transform:uppercase;}
.hero {padding:10px 0 22px;}
.hero h1 {font-size:clamp(2.5rem,5vw,4.2rem);line-height:1.08;letter-spacing:-.055em;margin:16px 0;color:#f5f3ff;}
.hero h1 span {color:#bba0ff;}
.hero p {max-width:650px;color:#aab5ce;font-size:1.08rem;line-height:1.7;}
.brand {font-size:1.2rem;font-weight:750;letter-spacing:-.04em;margin-bottom:8px;}
.brand span {color:#bba0ff;}
.side-note {color:#a7b2ca;font-size:.86rem;line-height:1.7;}
.results-heading {display:flex;justify-content:space-between;align-items:end;gap:20px;margin:32px 0 20px;}
.results-heading h2 {font-size:1.7rem;letter-spacing:-.035em;margin:0;color:#f1f0fa;}
.results-heading p {color:#aab5ce;font-size:.88rem;margin:5px 0 0;}
.count {color:#c8b5f5;border:1px solid #413559;background:#201d32;border-radius:999px;padding:8px 14px;white-space:nowrap;font-size:.8rem;}
.game-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;}
.game-card {background:#151c2c;border:1px solid #2c354d;border-radius:20px;overflow:hidden;transition:transform .18s,border-color .18s;}
.game-card:hover {transform:translateY(-3px);border-color:#8065b4;}
.game-card:first-child {grid-column:1/-1;display:grid;grid-template-columns:1.1fr 1fr;}
.artwork {aspect-ratio:460/215;background-size:cover;background-position:center;background-color:#292341;position:relative;min-height:190px;}
.game-card:first-child .artwork {aspect-ratio:auto;height:100%;min-height:320px;}
.rank {position:absolute;left:16px;top:16px;background:#0e1224e8;color:#ede6ff;border:1px solid #ffffff33;border-radius:9px;padding:7px 11px;font-size:.75rem;font-weight:700;}
.card-body {padding:24px;display:flex;flex-direction:column;gap:15px;}
.card-top {display:flex;align-items:start;justify-content:space-between;gap:16px;}
.card-top h3 {font-size:1.3rem;line-height:1.3;letter-spacing:-.025em;margin:0;color:#f5f4fc;}
.price {font-weight:750;color:#dac9ff;white-space:nowrap;}
.tags {display:flex;flex-wrap:wrap;gap:6px;}
.tag {font-size:.7rem;background:#232c40;border:1px solid #33405a;color:#c6d2e8;border-radius:6px;padding:4px 8px;}
.description {color:#b0bbd1;font-size:.9rem;line-height:1.65;margin:0;}
.card-bottom {margin-top:auto;display:flex;align-items:center;justify-content:space-between;gap:14px;padding-top:16px;border-top:1px solid #2b354d;}
.review-score {color:#8ce4c3;font-size:.95rem;font-weight:650;}
.review-count {font-size:.75rem;color:#a1acc5;margin-top:4px;}
.steam-link {display:inline-block;background:#b79af8;color:#171124!important;padding:11px 15px;border-radius:10px;text-decoration:none!important;font-size:.82rem;font-weight:750;white-space:nowrap;}
.steam-link:hover {background:#d1baff;}
.empty {padding:48px 24px;text-align:center;background:#151c2c;border:1px dashed #55456e;border-radius:18px;color:#c2c9dd;}
.footnote {color:#93a1bd;font-size:.78rem;line-height:1.7;margin-top:28px;}
@media(max-width:800px) {.game-grid{grid-template-columns:1fr}.game-card:first-child{display:block}.game-card:first-child .artwork{aspect-ratio:460/215;min-height:0}.results-heading{align-items:start;flex-direction:column}.card-body{padding:20px}.hero h1{font-size:2.65rem;}}
@media(prefers-reduced-motion:reduce){.game-card{transition:none}.game-card:hover{transform:none}}
</style>
"""
st.html(STYLE)
DATA_PATH = Path(__file__).resolve().parent / "data" / "games.csv"

@st.cache_data(show_spinner="Loading your game collection…")
def load_games(path, modified):
    # modified is a cache key: replacing the CSV causes a fresh load.
    with open(path, encoding="utf-8-sig", newline="") as file:
        headings = next(csv.reader(file))
    if "DiscountDLC count" in headings:
        position = headings.index("DiscountDLC count")
        headings[position:position + 1] = ["Discount", "DLC count"]
    needed = [
        "AppID", "Name", "Price", "Genres", "Positive", "Negative",
        "About the game", "Tags", "Categories"
    ]
    if "Header image" in headings:
        needed.append("Header image")
    data = pd.read_csv(path, header=0, names=headings, usecols=needed)
    data = data.dropna(subset=["Name"]).copy().reset_index(drop=True)
    if "Header image" not in data:
        data["Header image"] = ""
    data["Search text"] = (data["Genres"].fillna("") + " " + data["Tags"].fillna("") + " " + data["About the game"].fillna(""))
    return data

@st.cache_resource(show_spinner="Preparing text search — the first search takes a little longer…")
def build_search_index(texts):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    return vectorizer, vectorizer.fit_transform(texts)

def plain_text(value):
    """Make a short readable excerpt; HTML escaping happens at display time."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", str(value)))).strip()

def artwork_url(value):
    """Only use HTTPS Steam CDN artwork; otherwise show the gradient fallback."""
    if pd.isna(value):
        return ""
    value = str(value)
    parsed = urlparse(value)
    hosts = ("steamstatic.com", "steamcdn-a.akamaihd.net", "steamusercontent.com")
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not any(host == h or host.endswith("." + h) for h in hosts):
        return ""
    if any(c in value for c in "'\"()\\\n\r"):
        return ""
    return value

def card_html(game, rank):
    title = html.escape(str(game["Name"]))
    image = artwork_url(game["Header image"])
    background = (f"url('{html.escape(image, quote=True)}'), " if image else "") + "linear-gradient(135deg,#493669,#172e48)"
    price = "Free" if game["Price"] == 0 else f"${game['Price']:,.2f}"
    genre = "" if pd.isna(game["Genres"]) else str(game["Genres"])
    tags = "".join(f'<span class="tag">{html.escape(g.strip())}</span>' for g in genre.split(",")[:4] if g.strip())
    description = plain_text(game["About the game"])
    description = description[:210].rsplit(" ", 1)[0] + "…" if len(description) > 210 else description
    description = html.escape(description or "Explore this game on Steam for its description and details.")
    rating = game["Positive reviews (%)"]
    rating_text = "Not yet rated" if pd.isna(rating) else f"{rating:.1f}% positive"
    url = f"https://store.steampowered.com/app/{int(game['AppID'])}/"
    return f'''<article class="game-card">
    <div class="artwork" role="img" aria-label="Artwork for {title}" style="background-image:{background}"><span class="rank">{'TOP PICK' if rank == 1 else f'PICK {rank:02d}'}</span></div>
    <div class="card-body"><div class="card-top"><h3>{title}</h3><span class="price">{price}</span></div>
    <div class="tags">{tags}</div><p class="description">{description}</p>
    <div class="card-bottom"><div><div class="review-score">{rating_text}</div><div class="review-count">{int(game['Total reviews']):,} reviews</div></div>
    <a class="steam-link" href="{url}" target="_blank" rel="noopener noreferrer" aria-label="Open {title} on Steam">Open on Steam ↗</a></div></div></article>'''

if not DATA_PATH.exists():
    st.title("Your next game starts here")
    st.info("Add games.csv to the data folder next to app.py, then reload this page.")
    st.stop()
try:
    games = load_games(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
except (ValueError, OSError, pd.errors.ParserError) as error:
    st.error("The dataset could not be loaded. Check that data/games.csv is the extracted Steam dataset.")
    with st.expander("Technical details"):
        st.code(str(error))
    st.stop()

genre_lists = games["Genres"].fillna("").str.split(",")
genres = sorted({genre for values in genre_lists for genre in values if genre})

# Stable keys let the sidebar reset button restore the original defaults.
def reset_filters():
    st.session_state.update(
        genre="All genres",
        min_budget=0.0,
        budget=20.0,
        positive=0,
        reviews=100,
        play_style="Any play style"
    )

def set_query(text):
    st.session_state["query"] = text

with st.sidebar:
    st.html(
        '<div class="brand">◈ &nbsp; NEXT<span>GAME</span></div>'
        '<p class="side-note">Less searching. More playing.</p>'
    )

    st.subheader("Find your fit")

    selected_genre = st.selectbox(
        "Genre",
        ["All genres"] + genres,
        key="genre"
    )

    play_style = st.selectbox(
        "Play style",
        [
            "Any play style",
            "Single-player",
            "Multiplayer",
            "Co-op",
            "Online co-op",
            "Local co-op"
        ],
        key="play_style"
    )

    st.divider()
    st.caption("YOUR BUDGET · USD")

    price_left, price_right = st.columns(2)

    with price_left:
        min_price = st.number_input(
            "Minimum",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="min_budget"
        )

    with price_right:
        max_price = st.number_input(
            "Maximum",
            min_value=0.0,
            value=20.0,
            step=1.0,
            key="budget"
        )

    st.divider()
    st.caption("PLAYER REVIEWS")

    min_positive = st.number_input(
        "Minimum positive reviews (%)",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        key="positive"
    )

    min_reviews = st.number_input(
        "Minimum review count",
        min_value=0,
        value=100,
        step=100,
        key="reviews"
    )

    st.button(
        "Reset filters",
        on_click=reset_filters,
        use_container_width=True
    )

    st.divider()
    st.caption(f"{len(games):,} games to explore")
    st.caption("Prices reflect the dataset snapshot.")

st.html('<section class="hero"><div class="eyebrow">A little curiosity. A new adventure.</div><h1>Find your next<br><span>just one more hour.</span></h1><p>Tell us what you feel like playing. Discover five Steam games that fit your interests and your budget.</p></section>')
query = st.text_input("What would you like to play?", placeholder="A relaxing farming game with crafting…", key="query").strip()
with st.container(horizontal=True):
    for label, text in [("🌱 Cozy farming", "relaxing farming crafting"), ("🏛 Build a civilization", "civilization building strategy"), ("🚀 Explore space", "space exploration adventure")]:
        st.button(label, on_click=set_query, args=(text,))

# Same filters, rounding, and ranking as the original app.
filtered_games = games
play_style_categories = {
    "Single-player": {"Single-player"},
    "Multiplayer": {"Multi-player"},
    "Co-op": {
        "Co-op", "Online Co-op", "LAN Co-op",
        "Shared/Split Screen Co-op"
    },
    "Online co-op": {"Online Co-op"},
    "Local co-op": {"Shared/Split Screen Co-op"}
}

if play_style != "Any play style":
    allowed_categories = play_style_categories[play_style]

    category_sets = (
        filtered_games["Categories"]
        .fillna("")
        .str.split(",")
        .apply(lambda values: {value.strip() for value in values})
    )

    matches_style = category_sets.apply(
        lambda values: bool(values & allowed_categories)
    )

    filtered_games = filtered_games[matches_style]
if selected_genre != "All genres":
    genre_matches = (
        filtered_games["Genres"].fillna("").str.split(",")
        .apply(lambda values: selected_genre in values)
    )
    filtered_games = filtered_games[genre_matches]
if min_price > max_price:
    st.warning("Minimum price must be less than or equal to maximum price.")
    st.stop()

filtered_games = filtered_games[
    (filtered_games["Price"] >= min_price)
    & (filtered_games["Price"] <= max_price)
].copy()
filtered_games["Total reviews"] = filtered_games["Positive"] + filtered_games["Negative"]
review_totals = filtered_games["Total reviews"].replace(0, float("nan"))
filtered_games["Positive reviews (%)"] = (filtered_games["Positive"] / review_totals * 100).round(1)
if min_positive > 0:
    filtered_games = filtered_games[filtered_games["Positive reviews (%)"] >= min_positive]
filtered_games = filtered_games[filtered_games["Total reviews"] >= min_reviews]
if query:
    vectorizer, game_vectors = build_search_index(games["Search text"])
    scores = cosine_similarity(vectorizer.transform([query]), game_vectors).ravel()
    filtered_games = filtered_games.copy()
    filtered_games["Text similarity"] = scores[filtered_games.index]
    filtered_games = filtered_games[filtered_games["Text similarity"] > 0]
    recommendations = filtered_games.sort_values(by=["Text similarity", "Total reviews"], ascending=[False, False]).head(5)
else:
    recommendations = filtered_games.sort_values(by="Total reviews", ascending=False).head(5)

heading = "Picked for your search" if query else "Popular picks for you"
subtitle = "Ordered by word similarity, then review count." if query else "Ordered by review count. Add a search to make it personal."
st.html(f'<div class="results-heading"><div><h2>{heading}</h2><p>{subtitle}</p></div><span class="count">{len(filtered_games):,} matches · {len(recommendations)} picks</span></div>')
if recommendations.empty:
    st.html('<div class="empty"><h3>No games found this time</h3><p>Try broader search words, raise your budget, or lower the review requirements.</p></div>')
    st.button("Clear search", on_click=set_query, args=("",))
else:
    st.html('<div class="game-grid">' + "".join(card_html(game, rank) for rank, (_, game) in enumerate(recommendations.iterrows(), 1)) + '</div>')
    with st.expander("Compare the details"):
        comparison = recommendations.copy()
        comparison["Steam link"] = "https://store.steampowered.com/app/" + comparison["AppID"].astype(int).astype(str) + "/"
        columns = ["Name", "Price", "Genres", "Total reviews", "Positive reviews (%)"]
        if query:
            columns.append("Text similarity")
        st.dataframe(comparison[columns + ["Steam link"]], hide_index=True, column_config={"Steam link": st.column_config.LinkColumn("Steam store", display_text="Open on Steam")})
st.html('<p class="footnote">Prices are in USD from the dataset snapshot; current Steam prices and availability may differ. Text matching compares words, not meaning. Artwork loads from Steam’s CDN when available; otherwise a gradient is shown.</p>')
with st.expander("About this project"):
    st.write("Built for Artificial Intelligence and Data Analytics. The app combines reproducible data cleaning, preference filters, and TF-IDF text matching. Similarity is not a probability of enjoying a game.")
