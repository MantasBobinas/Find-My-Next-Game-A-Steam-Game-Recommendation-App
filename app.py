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
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

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
.why-game {background:#1d2438;border-left:3px solid #b79af8;border-radius:8px;padding:14px 16px;}
.why-game h4 {margin:0 0 8px;color:#dac9ff;font-size:.85rem;}
.why-game ul {margin:0;padding-left:18px;color:#bdc8df;font-size:.8rem;line-height:1.65;}
.card-bottom {margin-top:auto;display:flex;align-items:center;justify-content:space-between;gap:14px;padding-top:16px;border-top:1px solid #2b354d;}
.review-score {color:#8ce4c3;font-size:.95rem;font-weight:650;}
.review-count {font-size:.75rem;color:#a1acc5;margin-top:4px;}
.steam-link {display:inline-block;background:#b79af8;color:#171124!important;padding:11px 15px;border-radius:10px;text-decoration:none!important;font-size:.82rem;font-weight:750;white-space:nowrap;}
.steam-link:hover {background:#d1baff;}
.empty {padding:48px 24px;text-align:center;background:#151c2c;border:1px dashed #55456e;border-radius:18px;color:#c2c9dd;}
.footnote {color:#93a1bd;font-size:.78rem;line-height:1.7;margin-top:28px;}
@media(max-width:800px) {.game-grid{grid-template-columns:1fr}.game-card:first-child{display:block}.game-card:first-child .artwork{aspect-ratio:460/215;min-height:0}.results-heading{align-items:start;flex-direction:column}.card-body{padding:20px}.hero h1{font-size:2.65rem;}}
@media(prefers-reduced-motion:reduce){.game-card{transition:none}.game-card:hover{transform:none}}

.analysis-section{margin-top:50px;padding-top:30px;border-top:1px solid #30374d}
.analysis-section h2{color:#f5f3ff;font-size:2rem;letter-spacing:-.04em;margin:10px 0}
.analysis-intro{color:#aab5ce;line-height:1.7;margin:0 0 24px}
.analysis-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-bottom:24px}
.analysis-stat{padding:22px;background:linear-gradient(135deg,#29223f,#151c2c);border:1px solid #403455;border-radius:16px}
.analysis-stat-label{color:#c8bbdf;font-size:.78rem}
.analysis-stat-value{color:#f5f3ff;font-size:2rem;font-weight:750;margin:8px 0}
.analysis-stat-note,.analysis-note{color:#a8b2cc;font-size:.76rem;line-height:1.7}
.analysis-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}
.analysis-panel{background:#151c2c;border:1px solid #2c354d;border-radius:20px;padding:24px;min-width:0}
.analysis-panel.wide{grid-column:1/-1}
.analysis-panel h3{color:#f5f3ff;font-size:1.2rem;margin:0 0 8px}
.analysis-insight{color:#c9b5f4;font-size:.87rem;line-height:1.6;margin:0 0 22px}
.analysis-bars{display:flex;flex-direction:column;gap:13px}
.analysis-bar-label{display:flex;justify-content:space-between;gap:16px;color:#dbe1ef;font-size:.8rem;margin-bottom:6px}
.analysis-bar-count{font-variant-numeric:tabular-nums;color:#b9c6dc;white-space:nowrap}
.analysis-track{height:9px;background:#252e43;border-radius:999px;overflow:hidden}
.analysis-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#8262ce,#c4a6ff)}
.analysis-panel.price .analysis-fill{background:linear-gradient(90deg,#3d9c97,#8ce4c3)}
.analysis-panel.reviews .analysis-fill{background:linear-gradient(90deg,#4e83c6,#91bfff)}
.analysis-note{margin:18px 0 0}
@media(max-width:800px){.analysis-stats,.analysis-panels{grid-template-columns:1fr}.analysis-panel.wide{grid-column:auto}.analysis-panel{padding:20px}}

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

# These words describe the request, rather than the kind of game wanted.
SEARCH_STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | {
    "game", "games", "gaming", "looking", "look", "want", "wanted",
    "recommend", "recommended", "recommendation", "recommendations", "suggest",
    "suggestion", "suggestions", "please", "like", "play", "playing"
})


@st.cache_resource(show_spinner="Preparing text search — the first search takes a little longer…")
def build_search_index(metadata, descriptions):
    """Use one vocabulary, but score metadata and descriptions separately."""
    vectorizer = TfidfVectorizer(stop_words=SEARCH_STOP_WORDS, max_features=20000)
    vectorizer.fit(metadata + " " + descriptions)
    metadata_vectors = vectorizer.transform(metadata)
    description_vectors = vectorizer.transform(descriptions)
    # Each field is already normalized by TF-IDF. This gives genre/tag word
    # matches 75% of the score and description word matches 25%.
    game_vectors = 0.75 * metadata_vectors + 0.25 * description_vectors
    return vectorizer, game_vectors.tocsr(), metadata_vectors



@st.cache_data(show_spinner=False)
def analyse_dataset(data):
    """Summarise the full cleaned dataset, independently of search filters."""
    genre_lists = data["Genres"].fillna("").apply(
        lambda value: sorted({item.strip() for item in str(value).split(",") if item.strip()})
    )
    genre_counts = genre_lists.explode().dropna().value_counts().head(10)
    prices = pd.to_numeric(data["Price"], errors="coerce")
    valid_prices = prices[prices.notna() & (prices >= 0) & (prices < float("inf"))]
    price_labels = ["$0", "Over $0–$5", "Over $5–$10", "Over $10–$20",
                    "Over $20–$40", "Over $40–$60", "Over $60"]
    price_counts = pd.cut(
        valid_prices, bins=[-0.001, 0, 5, 10, 20, 40, 60, float("inf")],
        labels=price_labels
    ).value_counts(sort=False)
    positive = pd.to_numeric(data["Positive"], errors="coerce")
    negative = pd.to_numeric(data["Negative"], errors="coerce")
    totals = positive + negative
    valid_reviews = (positive >= 0) & (negative >= 0) & (totals < float("inf"))
    rated = valid_reviews & (totals > 0)
    percentages = positive[rated] / totals[rated] * 100
    review_labels = ["0–20%", "Over 20–40%", "Over 40–60%", "Over 60–80%", "Over 80–100%"]
    review_counts = pd.cut(
        percentages, bins=[-0.001, 20, 40, 60, 80, 100], labels=review_labels
    ).value_counts(sort=False)
    return {
        "genres": genre_counts.rename_axis("Genre").reset_index(name="Games"),
        "prices": price_counts.rename_axis("Price range").reset_index(name="Games"),
        "reviews": review_counts.rename_axis("Positive reviews").reset_index(name="Games"),
        "price_order": price_labels, "review_order": review_labels,
        "median_price": valid_prices.median(), "valid_prices": len(valid_prices),
        "zero_price": int((valid_prices == 0).sum()),
        "rated": int(rated.sum()),
        "unrated": int((valid_reviews & (totals == 0)).sum()),
        "invalid_reviews": int((~valid_reviews).sum()),
        "missing_genres": int(genre_lists.apply(len).eq(0).sum()),
    }


def analysis_bars(data, category):
    """Exact counts alongside bars; each chart uses its own scale."""
    maximum = int(data["Games"].max()) if not data.empty else 0
    if maximum == 0:
        return '<p class="analysis-note">No data available for this chart.</p>'
    rows = []
    for _, row in data.iterrows():
        label = html.escape(str(row[category]))
        count = int(row["Games"])
        width = count / maximum * 100
        rows.append(
            f'<div><div class="analysis-bar-label"><span>{label}</span>'
            f'<span class="analysis-bar-count">{count:,} games</span></div>'
            f'<div class="analysis-track" aria-hidden="true">'
            f'<div class="analysis-fill" style="width:{width:.3f}%"></div></div></div>'
        )
    return '<div class="analysis-bars">' + "".join(rows) + '</div>'


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

def matching_terms(vectorizer, game_vectors, query_vector, row_index):
    """Return up to six shared words, strongest TF-IDF contributions first."""
    # Multiplication keeps only words present in both the query and this game.
    contributions = game_vectors[row_index].multiply(query_vector).tocsr()
    features = vectorizer.get_feature_names_out()
    weighted_terms = sorted(
        zip(contributions.indices, contributions.data),
        key=lambda item: (-item[1], features[item[0]])
    )
    return [features[index] for index, weight in weighted_terms[:6] if weight > 0]


def explanation_html(game, terms, has_query, genre, style, low, high,
                     positive_threshold, review_threshold):
    """Describe matching evidence and filters without implying personal accuracy."""
    reasons = []
    if has_query:
        reasons.append("Top matching words: " + ", ".join(terms) + ".")
    else:
        reasons.append("Among the most-reviewed games that meet your filters.")
    if genre != "All genres":
        reasons.append(f"Matches your genre: {genre}.")
    if style != "Any play style":
        reasons.append(f"Listed play style matches: {style}.")
    reasons.append(
        f"Dataset price ${game['Price']:,.2f} fits your ${low:,.2f}–${high:,.2f} budget."
    )
    if positive_threshold > 0:
        reasons.append(
            f"{game['Positive reviews (%)']:.1f}% positive reviews meets your "
            f"{positive_threshold}% minimum."
        )
    if review_threshold > 0:
        reasons.append(
            f"{int(game['Total reviews']):,} reviews meets your "
            f"{review_threshold:,} minimum."
        )
    # Escape every reason because search and dataset text can contain HTML.
    items = "".join(f"<li>{html.escape(reason)}</li>" for reason in reasons)
    return '<section class="why-game"><h4>Why this game?</h4><ul>' + items + '</ul></section>'


def card_html(game, rank, explanation):
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
    {explanation}
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
require_metadata = st.checkbox(
    "Require a genre or tag match",
    value=True,
    help="For text searches, require at least one search word in a game's genres or tags. "
         "Turn this off to also discover games matching only their descriptions."
)
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
    metadata = games["Genres"].fillna("") + " " + games["Tags"].fillna("")
    descriptions = games["About the game"].fillna("")
    vectorizer, game_vectors, metadata_vectors = build_search_index(metadata, descriptions)
    query_vector = vectorizer.transform([query])
    scores = (game_vectors @ query_vector.T).toarray().ravel()
    if require_metadata:
        metadata_scores = (metadata_vectors @ query_vector.T).toarray().ravel()
        scores[metadata_scores == 0] = 0
    if query_vector.nnz == 0:
        st.info("Try specific interests such as racing, farming, or space exploration. "
                "Common request words such as 'looking' and 'game' are ignored; "
                "other words may not be in the search vocabulary.")
    filtered_games = filtered_games.copy()
    filtered_games["Text similarity"] = scores[filtered_games.index]
    filtered_games = filtered_games[filtered_games["Text similarity"] > 0]
    recommendations = filtered_games.sort_values(by=["Text similarity", "Total reviews"], ascending=[False, False]).head(5)
else:
    recommendations = filtered_games.sort_values(by="Total reviews", ascending=False).head(5)

heading = "Picked for your search" if query else "Popular picks for you"
subtitle = "Genre and tag matches carry more weight than description matches; review count breaks ties." if query else "Ordered by review count. Add a search to make it personal."
st.html(f'<div class="results-heading"><div><h2>{heading}</h2><p>{subtitle}</p></div><span class="count">{len(filtered_games):,} matches · {len(recommendations)} picks</span></div>')
if recommendations.empty:
    st.html('<div class="empty"><h3>No games found this time</h3><p>Try broader search words, raise your budget, or lower the review requirements.</p></div>')
    st.button("Clear search", on_click=set_query, args=("",))
else:
    cards = []
    for rank, (row_index, game) in enumerate(recommendations.iterrows(), 1):
        terms = matching_terms(vectorizer, game_vectors, query_vector, row_index) if query else []
        explanation = explanation_html(
            game, terms, bool(query), selected_genre, play_style,
            min_price, max_price, min_positive, min_reviews
        )
        cards.append(card_html(game, rank, explanation))
    st.html('<div class="game-grid">' + "".join(cards) + '</div>')
    with st.expander("Compare the details"):
        comparison = recommendations.copy()
        comparison["Steam link"] = "https://store.steampowered.com/app/" + comparison["AppID"].astype(int).astype(str) + "/"
        columns = ["Name", "Price", "Genres", "Total reviews", "Positive reviews (%)"]
        if query:
            columns.append("Text similarity")
        st.dataframe(comparison[columns + ["Steam link"]], hide_index=True, column_config={"Steam link": st.column_config.LinkColumn("Steam store", display_text="Open on Steam")})

# Always visible below recommendations; summaries use the full dataset.
summary = analyse_dataset(games)
median = summary["median_price"]
median_label = "N/A" if pd.isna(median) else f"${median:,.2f}"
rated_share = summary["rated"] / len(games) * 100 if len(games) else 0
if summary["genres"].empty:
    genre_insight = "Explore how games are categorised in this dataset."
else:
    leader = summary["genres"].iloc[0]
    genre_insight = f"{leader['Genre']} is the most common genre label, appearing on {int(leader['Games']):,} games."
cheap_count = int(summary["prices"]["Games"].iloc[:2].sum())
cheap_share = cheap_count / summary["valid_prices"] * 100 if summary["valid_prices"] else 0
positive_count = int(summary["reviews"]["Games"].iloc[-1])
positive_share = positive_count / summary["rated"] * 100 if summary["rated"] else 0
price_insight = f"{cheap_share:.1f}% of valid listed prices are $5 or less, including $0 entries." if summary["valid_prices"] else "No valid prices are available."
review_insight = f"{positive_share:.1f}% of reviewed games have more than 80% positive reviews." if summary["rated"] else "No games with valid reviews are available."
st.html(f'''<section class="analysis-section" aria-label="Dataset analysis">
<div class="eyebrow">Behind the recommendations</div>
<h2>A closer look at the games</h2>
<p class="analysis-intro">Explore the full dataset: genres, prices and player reviews. Search and sidebar filters do not change these figures.</p>
<div class="analysis-stats">
<div class="analysis-stat"><div class="analysis-stat-label">GAMES TO EXPLORE</div><div class="analysis-stat-value">{len(games):,}</div><div class="analysis-stat-note">Games in the cleaned dataset</div></div>
<div class="analysis-stat"><div class="analysis-stat-label">MEDIAN LISTED PRICE</div><div class="analysis-stat-value">{median_label}</div><div class="analysis-stat-note">Middle price in USD, including $0 entries</div></div>
<div class="analysis-stat"><div class="analysis-stat-label">GAMES WITH REVIEWS</div><div class="analysis-stat-value">{summary['rated']:,}</div><div class="analysis-stat-note">{rated_share:.1f}% of the cleaned dataset</div></div>
</div><div class="analysis-panels">
<section class="analysis-panel wide"><h3>The 10 most common genres</h3><p class="analysis-insight">{html.escape(genre_insight)}</p>
{analysis_bars(summary['genres'], 'Genre')}
<p class="analysis-note">Games may appear in several genres and are counted once in each. {summary['missing_genres']:,} games have no listed genre. Counts describe the catalogue, not player popularity.</p></section>
<section class="analysis-panel price"><h3>What does your next game cost?</h3><p class="analysis-insight">{price_insight}</p>
{analysis_bars(summary['prices'], 'Price range')}
<p class="analysis-note">USD prices from the dataset snapshot. {summary['zero_price']:,} games are listed at $0; they may not be free today. {len(games) - summary['valid_prices']:,} missing or invalid prices are excluded.</p></section>
<section class="analysis-panel reviews"><h3>How do players rate the games?</h3><p class="analysis-insight">{review_insight}</p>
{analysis_bars(summary['reviews'], 'Positive reviews')}
<p class="analysis-note">Positive percentage = positive reviews ÷ total reviews × 100. Each game counts once. Excludes {summary['unrated']:,} games without reviews and {summary['invalid_reviews']:,} with missing or invalid counts. Small review counts can produce less reliable percentages.</p></section>
</div><p class="analysis-note">Bar lengths are scaled within each chart; use the labels to compare exact counts across charts. These patterns describe the dataset and do not measure recommendation accuracy or explain what causes popularity.</p>
</section>''')

st.html('<p class="footnote">Prices are in USD from the dataset snapshot; current Steam prices and availability may differ. Text matching compares words, not meaning. Common request words are ignored. Artwork loads from Steam’s CDN when available; otherwise a gradient is shown.</p>')
with st.expander("About this project"):
    st.write("Built for Artificial Intelligence and Data Analytics. The app combines reproducible data cleaning, preference filters, and TF-IDF text matching. Search weights genre/tag matches at 75% and description matches at 25%; these are design choices, not measured accuracy. Similarity is not a probability of enjoying a game.")
