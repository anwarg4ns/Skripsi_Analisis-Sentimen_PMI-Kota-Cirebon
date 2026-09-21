import os
from pathlib import Path

import joblib
import mysql.connector
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

from preprocessing import preprocess_text


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "models" / "nb_model.pkl"
VECTORIZER_FILE = PROJECT_DIR / "models" / "tfidf_vectorizer.pkl"
DATABASE_NAME = "db_pmi_cirebon"
TABLE_NAME = "tabel_ulasan"

DB_CONFIG = {
    "host": os.getenv("PMI_DB_HOST", "localhost"),
    "port": int(os.getenv("PMI_DB_PORT", "3306")),
    "user": os.getenv("PMI_DB_USER", "root"),
    "password": os.getenv("PMI_DB_PASSWORD", ""),
    "database": DATABASE_NAME,
}


st.set_page_config(
    page_title="Sentimen PMI",
    page_icon=":material/health_and_safety:",
    layout="wide",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Poppins', sans-serif;
            font-size: 17px;
        }

        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stForm"],
        [data-testid="stTextArea"] textarea,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stButton"] button,
        [data-testid="stAlert"] {
            border-radius: 10px;
        }

        [data-testid="stMetric"] {
            padding: 16px;
        }

        [data-testid="stDataFrame"], [data-testid="stForm"] {
            overflow: hidden;
        }

        [data-testid="stTextArea"] textarea {
            font-size: 17px;
            line-height: 1.55;
            padding: 14px;
        }

        [data-testid="stFormSubmitButton"] button,
        [data-testid="stButton"] button {
            font-size: 16px;
            font-weight: 600;
            min-height: 44px;
            transition: transform 0.2s ease;
        }

        [data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stButton"] button:hover {
            transform: translateY(-1px);
        }

        [data-testid="stAlert"] {
            font-size: 17px;
        }

        .review-table {
            border-collapse: collapse;
            min-width: 1200px;
            width: 100%;
            font-size: 16px;
        }

        .review-table-container {
            max-height: 500px;
            overflow-x: auto;
            overflow-y: auto;
            border: 1px solid #444;
            border-radius: 10px;
        }

        .review-table th {
            font-size: 18px;
            font-weight: 700;
            text-align: center !important;
            padding: 12px 15px;
            position: sticky !important;
            top: 0 !important;
            background-color: #f0f2f6 !important;
            background-color: var(--secondary-background-color, #f0f2f6) !important;
            color: #1e1e1e !important;
            color: var(--text-color, #1e1e1e) !important;
            z-index: 9999 !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
        }

        .review-table td {
            vertical-align: middle;
            padding: 12px 15px;
            border-bottom: 1px solid #555;
            text-align: center;
        }

        .review-table th:nth-child(1),
        .review-table td:nth-child(1),
        .review-table th:nth-child(2),
        .review-table td:nth-child(2),
        .review-table th:nth-child(4),
        .review-table td:nth-child(4) {
            white-space: nowrap;
        }

        .review-table th:nth-child(3),
        .review-table td:nth-child(3),
        .review-table th:nth-child(5),
        .review-table td:nth-child(5) {
            min-width: 300px;
            white-space: normal;
            text-align: left;
        }

        .review-table tr:hover {
            background-color: rgba(255, 255, 255, 0.05);
        }

        [data-testid="stSidebar"] button {
            border: none !important;
            background: transparent !important;
            font-size: 26px !important;
            font-weight: 900 !important;
            color: var(--text-color) !important;
            white-space: normal !important;
            text-align: center !important;
            line-height: 1.3 !important;
        }

        [data-testid="stSidebar"] button p {
            font-size: 24px !important;
            font-weight: 900 !important;
            color: var(--text-color) !important;
            white-space: normal !important;
            text-align: center !important;
            line-height: 1.3 !important;
        }

        [data-testid="stSidebar"] button:hover,
        [data-testid="stSidebar"] button:hover p {
            color: #ff4b4b !important;
            background: transparent !important;
        }

        @media (prefers-color-scheme: dark) {
            .review-table th {
                background-color: #262730 !important;
                color: #ffffff !important;
            }

        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl="10m", max_entries=1)
def load_dataset() -> pd.DataFrame:
    """Load the thesis dataset from MySQL and cache it briefly for the session."""
    connection = mysql.connector.connect(**DB_CONFIG)
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            f"SELECT id, tanggal, ulasan, sentimen, ulasan_bersih "
            f"FROM `{TABLE_NAME}` ORDER BY id"
        )
        dataframe = pd.DataFrame(cursor.fetchall())
    finally:
        cursor.close()
        connection.close()

    dataframe["ulasan_bersih"] = dataframe["ulasan_bersih"].fillna("")
    return dataframe


@st.cache_resource
def load_model_artifacts():
    """Load the SMOTE-trained Naive Bayes model and its fitted vectorizer."""
    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            "Model atau vectorizer tidak ditemukan. Jalankan model_training.py terlebih dahulu."
        )

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    return model, vectorizer


def render_kpi_cards(dataframe: pd.DataFrame) -> None:
    """Render the four executive summary cards required by the dashboard."""
    counts = dataframe["sentimen"].value_counts()
    metrics = [
        ("Total ulasan", len(dataframe)),
        ("Positif", int(counts.get("Positif", 0))),
        ("Netral", int(counts.get("Netral", 0))),
        ("Negatif", int(counts.get("Negatif", 0))),
    ]

    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.metric(label, f"{value:,}", border=True, help=f"Jumlah {label.lower()}")


def render_review_table(dataframe: pd.DataFrame) -> None:
    """Render review data inside a fixed-height scrollable table container."""
    table_html = dataframe.to_html(
        classes="review-table table table-striped",
        index=False,
        border=0,
    )
    st.markdown(
        f"""
        <div class="review-table-container">
            {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_home(dataframe: pd.DataFrame) -> None:
    st.title("Sistem Analisis Sentimen PMI Kota Cirebon", icon=":material/health_and_safety:")

    render_kpi_cards(dataframe)

    st.subheader("Data ulasan", icon=":material/table_chart:")
    render_review_table(dataframe)


def show_single_prediction() -> None:
    st.title("Pengujian sentimen", icon=":material/psychology:")
    st.markdown(
        "<h4>Masukkan umpan balik masyarakat untuk memperoleh klasifikasi sentimen.</h4>",
        unsafe_allow_html=True,
    )

    with st.form(key="predict_form", border=True):
        review = st.text_area(
            "Ulasan masyarakat",
            height=160,
            placeholder="Contoh: Pelayanan sangat cepat dan petugasnya ramah.",
        )
        submitted = st.form_submit_button(
            "Prediksi sentimen",
            type="primary",
            icon=":material/analytics:",
        )

    if not submitted:
        return

    if not review.strip():
        st.warning("Masukkan ulasan terlebih dahulu.", icon=":material/warning:")
        return

    processed_review = preprocess_text(review)
    if not processed_review.strip():
        st.warning(
            "Ulasan tidak memiliki kata yang dapat dianalisis setelah preprocessing.",
            icon=":material/filter_alt:",
        )
        return

    model, vectorizer = load_model_artifacts()
    features = vectorizer.transform([processed_review])
    prediction = model.predict(features)[0]

    st.subheader("Hasil prediksi", icon=":material/insights:")
    if prediction == "Positif":
        st.success(f"Sentimen terdeteksi: **{prediction}**", icon=":material/thumb_up:")
    elif prediction == "Negatif":
        st.error(f"Sentimen terdeteksi: **{prediction}**", icon=":material/thumb_down:")
    else:
        st.warning(f"Sentimen terdeteksi: **{prediction}**", icon=":material/remove:")

    with st.expander("Lihat hasil preprocessing", icon=":material/text_fields:"):
        st.write(processed_review)


def show_dashboard(dataframe: pd.DataFrame) -> None:
    st.title("Visualisasi Sentimen", icon=":material/dashboard:")
    st.caption("Distribusi sentimen untuk mendukung pemantauan kualitas pelayanan.")
    render_kpi_cards(dataframe)

    sentiment_counts = (
        dataframe["sentimen"]
        .value_counts()
        .rename_axis("Sentimen")
        .reset_index(name="Jumlah")
    )
    sentiment_colors = {
        "Positif": "#28a745",
        "Negatif": "#dc3545",
        "Netral": "#6c757d",
    }
    chart_font = {
        "family": "Poppins, sans-serif",
        "size": 18,
    }

    chart_columns = st.columns(2)
    with chart_columns[0]:
        with st.container(border=True):
            st.subheader("Persentase Sentimen", icon=":material/pie_chart:")
            pie_chart = px.pie(
                sentiment_counts,
                names="Sentimen",
                values="Jumlah",
                hole=0.42,
                color="Sentimen",
                color_discrete_map=sentiment_colors,
            )
            pie_chart.update_traces(
                textinfo="percent",
                textfont=dict(size=20),
            )
            pie_chart.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                font=chart_font,
                legend=dict(font=dict(size=18)),
            )
            st.plotly_chart(pie_chart, width="stretch", theme="streamlit")

    with chart_columns[1]:
        with st.container(border=True):
            st.subheader("Jumlah ulasan", icon=":material/bar_chart:")
            bar_chart = px.bar(
                sentiment_counts,
                x="Sentimen",
                y="Jumlah",
                text_auto=True,
                color="Sentimen",
                color_discrete_map=sentiment_colors,
            )
            bar_chart.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                font=chart_font,
                showlegend=False,
                xaxis=dict(
                    title=dict(text="Sentimen", font=dict(size=19)),
                    tickfont=dict(size=17),
                ),
                yaxis=dict(
                    title=dict(text="Jumlah", font=dict(size=19)),
                    tickfont=dict(size=17),
                ),
            )
            st.plotly_chart(bar_chart, width="stretch", theme="streamlit")


def show_complaints(dataframe: pd.DataFrame) -> None:
    st.title("Analisis keluhan", icon=":material/report_problem:")
    st.markdown(
        "<h4>Daftar ulasan negatif untuk membantu PMI memprioritaskan evaluasi operasional.</h4>",
        unsafe_allow_html=True,
    )

    complaints = dataframe[dataframe["sentimen"] == "Negatif"].copy()
    st.metric(
        "Total keluhan",
        f"{len(complaints):,}",
        border=True,
        help="Jumlah ulasan dengan label Negatif",
    )
    render_review_table(complaints)


def main() -> None:
    menu_options = [
        "Dashboard",
        "Pengujian Sentimen",
        "Visualisasi Sentimen",
        "Analisis Keluhan",
    ]
    if "active_menu" not in st.session_state:
        st.session_state.active_menu = "Dashboard"

    with st.sidebar:
        if st.sidebar.button(
            "Sistem Analisis Sentimen PMI Kota Cirebon",
            key="header_btn",
        ):
            st.session_state.active_menu = "Dashboard"
            st.rerun()

        try:
            default_idx = menu_options.index(st.session_state.active_menu)
        except ValueError:
            default_idx = 0

        selected_page = option_menu(
            menu_title="",
            options=menu_options,
            icons=["house", "chat-square-text", "bar-chart-line", "exclamation-triangle"],
            menu_icon="list",
            default_index=default_idx,
        )

        if selected_page != st.session_state.active_menu:
            st.session_state.active_menu = selected_page
            st.rerun()

    try:
        dataframe = load_dataset()
    except (FileNotFoundError, KeyError, mysql.connector.Error) as error:
        st.error(f"Gagal memuat data dari MySQL: {error}", icon=":material/error:")
        st.stop()

    if selected_page == "Dashboard":
        show_home(dataframe)
    elif selected_page == "Pengujian Sentimen":
        show_single_prediction()
    elif selected_page == "Visualisasi Sentimen":
        show_dashboard(dataframe)
    else:
        show_complaints(dataframe)


if __name__ == "__main__":
    main()
