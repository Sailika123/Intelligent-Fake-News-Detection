from __future__ import annotations

# isort: skip_file
#import - required tools

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from core.inference import Prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Finding project folder. Locate the project root so the local modules can be imported.
here = Path(__file__).resolve().parent
project = None

for folder in [here, *here.parents]:
    if (folder / "config.py").exists():
        project = folder
        break

if project is None:
    st.error("config.py was not found.")
    st.stop()

sys.path.insert(0, str(project))
sys.path.insert(0, str(here))

#iporting own project files

import auth
import config

# setting up streamlit 

st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
)

# CSS - making app look better
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1040px;
            padding: 2.5rem 3rem 4rem;
        }

        .verdict-card {
            margin: 1rem 0;
            padding: 1.1rem 1.25rem;
            border-radius: 8px;
        }

        .verdict-real {
            background: rgba(33, 195, 84, 0.10);
            border: 1px solid rgba(33, 195, 84, 0.30);
        }

        .verdict-fake {
            background: rgba(255, 75, 75, 0.10);
            border: 1px solid rgba(255, 75, 75, 0.30);
        }

        .v-label {
            margin: 0;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .v-sub {
            margin: 0.35rem 0 0;
            opacity: 0.8;
        }

        .model-note {
            margin-top: 1rem;
            padding: 0.85rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 8px;
            font-size: 0.9rem;
            opacity: 0.85;
        }

        .page-intro {
            margin-bottom: 1.5rem;
        }

        .page-intro p {
            color: rgba(128, 128, 128, 0.95);
            margin: 0.35rem 0 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# Login system -- Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_page():
    """Show the sign-in and account creation page."""

    st.title("Fake News Detector")
    st.write(
        "Sign in to check articles, upload files, or test a public news URL."
    )

    _, center, _ = st.columns([1, 3, 1])

    with center:
        login_tab, signup_tab = st.tabs(
            ["Sign in", "Create account"]
        )

        with login_tab, st.form("login"):
            st.subheader("Welcome back")

            username = st.text_input("Username")
            password = st.text_input(
                "Password",
                type="password",
            )

            if st.form_submit_button(
                "Sign in",
                type="primary",
                use_container_width=True,
            ):
                if auth.verify(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with signup_tab, st.form("signup"):
            st.subheader("New account")

            new_username = st.text_input(
                "Choose a username",
                key="new_username",
            )
            st.caption(
                "Usernames can contain letters, numbers, spaces, "
                "symbols, underscores, hyphens, and other characters."
            )

            new_display_name = st.text_input(
                "Display name",
                key="new_display_name",
            )
            new_password = st.text_input(
                "Password",
                type="password",
                key="new_password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password",
            )

            if st.form_submit_button(
                "Create account",
                use_container_width=True,
            ):
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    created, message = auth.create_user(
                        new_username,
                        new_password,
                        new_display_name,
                    )
                    if created:
                        st.success(message + " You can sign in now.")
                    else:
                        st.error(message)

auth.ensure_default_user()

if not st.session_state.authenticated:   #if user not logged in page shows stop
    login_page()
    st.stop()

#Loading actual  Model
@st.cache_resource(show_spinner="Loading model...")
def get_model() -> tuple[Any, Any]:
    from core.inference import load_inference_bundle
    #find trained bilstm model
    model_path = (
        Path(config.ARTIFACT_DIR)
        / "bilstm_cnn_attention.keras"
    )
    #find saved tokenizer
    tokenizer_path = (
        Path(config.ARTIFACT_DIR)
        / "tokenizer.pkl"
    )

    return load_inference_bundle(
        str(model_path),
        str(tokenizer_path),
    )

try:
    model, tokenizer = get_model()
except (FileNotFoundError, OSError, ValueError, ImportError) as error:
    st.error("The trained model could not be loaded.")
    st.code(str(error))
    st.stop()
    
# Helper functions
def analyze(title: str, body: str) -> Prediction:
    """Run the same prediction pipeline used by manual input and URL input."""

    from core.inference import predict

    return predict(
        title,
        body,
        model,
        tokenizer,
        config.MAX_LEN,
    )
    
#url extraction
def fetch_article_safely(url: str) -> tuple[bool, str, str, str]:
    """
    Fetch an article using the project's URL extractor.

    The extractor should return:
        success, headline, article_body, error_message

    A small fallback using trafilatura is included so the UI can
    still handle websites whose HTML structure is difficult to parse.
    """

    from core.url_fetcher import fetch_article

    try:
        result = cast(tuple[Any, ...], fetch_article(url))

        # New extractor format.
        if len(result) == 4:
            success, title, body, error_message = result

            return (
                bool(success),
                str(title or "").strip(),
                str(body or "").strip(),
                str(error_message or "").strip(),
            )

        # Backward compatibility with the older two-value extractor.
        if len(result) == 2:
            title, body = result

            title = str(title or "").strip()
            body = str(body or "").strip()

            if len(body.split()) >= 80:
                return True, title, body, ""

            return (
                False,
                title,
                "",
                "The main article content could not be extracted reliably.",
            )

        raise ValueError(
            "The URL extractor returned an unsupported result format."
        )

    except (OSError, ValueError, TypeError) as first_error:
        logger.warning(
            "Primary URL extraction failed: %s",
            first_error,
        )

        # Optional fallback for difficult news sites.
        try:
            import trafilatura

            downloaded = trafilatura.fetch_url(url)

            if downloaded:
                extracted = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    include_links=False,
                    favor_precision=True,
                    output_format="txt",
                )

                metadata = trafilatura.extract_metadata(
                    downloaded
                )

                title = str(
                    getattr(metadata, "title", "") or ""
                ).strip()

                extracted = str(
                    extracted or ""
                ).strip()

                if extracted and len(extracted.split()) >= 80:
                    return True, title, extracted, ""

        except (OSError, ValueError, TypeError) as fallback_error:
            logger.warning(
                "Secondary URL extraction failed: %s",
                fallback_error,
            )

        return (
            False,
            "",
            "",
            "The main article content could not be extracted reliably.",
        )

#how many letters in headline are upper
def title_caps_ratio(text: str) -> float:
    letters = [x for x in text if x.isalpha()]

    if not letters:
        return 0.0

    upper = sum(1 for x in letters if x.isupper())
    return upper / len(letters)

#controls how prediction appears
def show_prediction(result: Prediction) -> None:
    """Display the model result without presenting it as a fact check."""

    label = result["label"]
    confidence = float(result["confidence"])

    if label == "Real":
        st.markdown(
            f"""
            <div class="verdict-card verdict-real">
                <p class="v-label">Predicted: Real</p>
                <p class="v-sub">Confidence: {confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif label == "Fake":
        st.markdown(
            f"""
            <div class="verdict-card verdict-fake">
                <p class="v-label">Predicted: Fake</p>
                <p class="v-sub">Confidence: {confidence:.1%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.info(
            f"Predicted: {label}. The result is close to the decision boundary."
        )

    if confidence < 0.65:
        st.warning(
            "The model is not very confident. Please verify the information "
            "with another reliable source."
        )

    st.markdown(
        """
        <div class="model-note">
            This is a machine-learning prediction based on patterns learned
            during model training. It is not independent fact-checking and
            should not be treated as proof that an article is definitely real
            or fake.
        </div>
        """,
        unsafe_allow_html=True,
    )
# Sidebar lefthand sidebar
with st.sidebar:
    hour = datetime.now().astimezone().hour

    if hour < 12:
        greeting = "Good morning"
    elif hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    username = st.session_state.get("username", "")

    st.write(
        f"**{greeting}, {auth.display_name(username)}**"
    )

    if st.button("Log out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.pop("username", None)
        st.rerun()

    st.divider()
    st.write("**Model:** BiLSTM-CNN-Attention")
    
# Main page
st.title("Fake News Detector")
st.markdown(
    """
    <div class="page-intro">
        <p>Review an article, a public URL, or a small batch of files.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

manual, url_tab, upload = st.tabs(
    ["Manual Input", "Check URL", "File Upload"]
)

# Manual article
with manual:
    title = st.text_input(
        "Headline",
        placeholder="Enter the headline...",
    )

    body = st.text_area(
        "Article body",
        height=220,
        placeholder="Paste the article here...",
    )

    if st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    ):
        if not title.strip() and not body.strip():
            st.warning("Please enter a headline or article body.")
        else:
            with st.spinner("Analyzing..."):
                result = analyze(title, body)

            show_prediction(result)

            with st.expander("Article details"):
                col1, col2 = st.columns(2)

                col1.metric(
                    "Title ALL-CAPS",
                    f"{title_caps_ratio(title):.0%}",
                )

                col2.metric(
                    "Exclamation marks",
                    title.count("!"),
                )

# Public news URL
with url_tab:
    st.write(
        "Paste a public news article URL. The application will "
        "keep the headline and main article content while "
        "excluding common webpage navigation and unrelated content."
    )

    article_url = st.text_input(
        "Article URL",
        placeholder="https://example.com/news/article",
    )

    if st.button(
        "Fetch and Analyze URL",
        type="primary",
        use_container_width=True,
    ):
        if not article_url.strip():
            st.warning("Please enter a URL first.")
        else:
            with st.spinner(
                "Fetching and extracting the main article..."
            ):
                success, url_title, url_body, error_message = (
                    fetch_article_safely(article_url.strip())
                )

            if not success:
                st.error(
                    error_message
                    or "The main article content could not be extracted."
                )

                st.info(
                    "The model was not run because the page content "
                    "could not be separated into a reliable headline "
                    "and main article."
                )

            else:
                word_count = len(url_body.split())

                st.success(
                    "The headline and main article content were extracted."
                )

                st.text_input(
                    "Article headline",
                    value=url_title,
                    disabled=True,
                )

                st.text_area(
                    "Main article content",
                    value=url_body,
                    height=300,
                    disabled=True,
                )

                st.caption(
                    f"Clean article length: {word_count:,} words"
                )

                if word_count < 80:
                    st.warning(
                        "The extracted article is too short for a reliable "
                        "model prediction."
                    )
                else:
                    with st.spinner(
                        "Analyzing the main article content..."
                    ):
                        result = analyze(
                            url_title,
                            url_body,
                        )

                    show_prediction(result)
# File upload
with upload:
    st.write(
        "Use a TXT file for one article or a CSV file for multiple articles."
    )

    file = st.file_uploader(
        "Choose file",
        type=["txt", "csv"],
    )

    if file is not None:
        # TXT
        if file.name.lower().endswith(".txt"):
            text = (
                file.read()
                .decode("utf-8", errors="replace")
                .strip()
            )

            parts = text.split("\n", 1)

            file_title = parts[0].strip() if parts else ""
            file_body = parts[1].strip() if len(parts) > 1 else ""

            st.text_input(
                "Detected headline",
                file_title,
                disabled=True,
            )

            st.text_area(
                "Detected body",
                file_body,
                height=180,
                disabled=True,
            )

            if st.button(
                "Analyze Document",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analyzing..."):
                    result = analyze(file_title, file_body)

                show_prediction(result)

        # CSV
        else:
            try:
                data = pd.read_csv(file)
            except (
                pd.errors.ParserError,
                UnicodeDecodeError,
                ValueError,
            ) as error:
                st.error(f"Could not read CSV: {error}")
                st.stop()

            if "text" not in data.columns:
                st.error("The CSV must contain a 'text' column.")
                st.stop()

            if "title" not in data.columns:
                data["title"] = ""

            st.dataframe(  # pyright: ignore[reportUnknownMemberType]
                cast(Any, data[["title", "text"]].head()),
                use_container_width=True,
            )

            if data.empty:
                st.warning("The CSV has no rows.")
                st.stop()

            maximum = min(len(data), 500)
            default = min(len(data), 50)

            rows = st.slider(
                "Rows to analyze",
                1,
                maximum,
                default,
            )

            if st.button(
                "Run Batch",
                type="primary",
                use_container_width=True,
            ):
                batch = data.head(rows).copy()

                progress = st.progress(
                    0,
                    text="Analyzing articles...",
                )

                predictions: list[str] = []
                confidence: list[float] = []
                real_probability: list[float] = []
                                                      
                for number, row in enumerate(
                    batch.itertuples(index=False),
                    start=1,
                ):
                    result = analyze(
                        str(getattr(row, "title", "")),
                        str(row.text),
                    )

                    predictions.append(result["label"])
                    confidence.append(result["confidence"])
                    real_probability.append(result["probability_real"])

                    progress.progress(
                        number / rows,
                        text=f"Article {number}/{rows}",
                    )

                progress.empty()

                batch["prediction"] = predictions
                batch["confidence"] = confidence
                batch["prob_real"] = real_probability

                st.success(
                    f"Finished {len(batch)} articles."
                )

                st.dataframe(  # pyright: ignore[reportUnknownMemberType]
                    cast(Any, batch),
                    use_container_width=True,
                )

                st.download_button(
                    "Download Predictions",
                    batch.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )