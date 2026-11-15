import io
import time
import urllib3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# MECH ORBIT - SPHEREx SKY EXPLORER
# ==========================================

SIA_URL = "https://caltech.edu"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

# ==========================================
# SESSION STATE
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = []

# ==========================================
# HTTP SESSION WITH RETRIES
# ==========================================

def make_session():
    session = requests.Session()
    session.verify = False

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0",
        "Accept": "text/csv"
    })

    return session

# ==========================================
# ARCHIVE SEARCH (WORKING CSV + COLUMN FIX)
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = make_session()

    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}",
        "RESPONSEFORMAT": "CSV"
    }

    response = session.get(
        SIA_URL,
        params=params,
        timeout=(20, 120)
    )

    response.raise_for_status()

    if not response.content:
        raise ValueError("Empty response from NASA archive.")

    content = response.content.lstrip().lower()

    if content.startswith(b"<html") or b"proxy error" in content[:3000]:
        raise ValueError("NASA archive returned an HTML/proxy error.")

    # मघाशी ज्या पद्धतीने ३४८ ऑब्झर्व्हेशन्स आले होते तीच पद्धत
    df = pd.read_csv(io.BytesIO(response.content), on_bad_lines="skip")
    df.columns = [str(c).strip().lower() for c in df.columns]

    # जर index_col मुळे कॉलम्स सरकले असतील तर ते आपोआप सरळ करणे
    if "s_ra" in df.columns and df["s_ra"].astype(str).str.contains("SPHEREx", case=False, na=False).any():
        df = df.reset_index()
        # मूळ कॉलम्सची नावे योग्य जागी बसवणे
        orig_cols = list(df.columns[1:])
        df = df.iloc[:, :len(orig_cols)]
        df.columns = orig_cols

    df["selected_collection"] = collection
    return df

# ==========================================
# FITS IMAGE DOWNLOAD
# ==========================================

@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    session = make_session()

    response = session.get(
        url,
        timeout=(20, 180)
    )

    response.raise_for_status()

    with fits.open(io.BytesIO(response.content), memmap=False) as hdus:
        for hdu in hdus:
            data = getattr(hdu, "data", None)

            if data is None:
                continue

            data = np.asarray(data)

            if data.ndim < 2:
                continue

            while data.ndim > 2:
                data = data[data.shape // 2]

            if data.ndim == 2 and np.isfinite(data).any():
                return data.astype(np.float32), hdu.name

    raise ValueError("No valid 2D image found in FITS file.")

# ==========================================
# IMAGE DISPLAY
# ==========================================

def show_image(url, heading):
    st.markdown(f"### {heading}")

    try:
        with st.spinner("Loading infrared image..."):
            image, extension = load_fits(url)

        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, (1, 99.5))

        if low == high:
            low = float(finite.min())
            high = float(finite.max()) + 1e-6

        fig, ax = plt.subplots(figsize=(6, 5))

        norm = ImageNormalize(
            vmin=low,
            vmax=high,
            stretch=AsinhStretch()
        )

        ax.imshow(
            np.ma.masked_invalid(image),
            origin="lower",
            cmap="magma",
            norm=norm
        )

        ax.set_title(heading)
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.caption(
            f"Extension: {extension} | "
            f"Size: {image.shape} × {image.shape} pixels"
        )

        st.link_button(
            "Open Original FITS",
            url,
            use_container_width=True
        )

    except Exception as error:
        st.warning(f"Image unavailable: {error}")

# ==========================================
# HEADER
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")

st.write(
    "Explore real infrared observations from "
    "NASA/IPAC Infrared Science Archive."
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("🔭 Search the Sky")

    ra = st.number_input(
        "Right Ascension (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=180.00,
        step=0.1,
        format="%.2f"
    )

    dec = st.number_input(
        "Declination (degrees)",
        min_value=-90.0,
        max_value=90.0,
        value=0.00,
        step=0.1,
        format="%.2f"
    )

    radius = st.number_input(
        "Search Radius (degrees)",
        min_value=0.01,
        max_value=5.0,
        value=0.05,
        step=0.05,
        format="%.2f"
    )

    collections = st.multiselect(
        "SPHEREx Collections",
        options=("spherex_dr1", "spherex_qr3"),
        default=("spherex_dr1", "spherex_qr3")
    )

    search_btn = st.button("🚀 Search NASA Archive", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Results", use_container_width=True)

    st.caption("Source: NASA/IPAC IRSA")

if clear_btn:
    st.cache_data.clear()
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []
    st.rerun()

# ==========================================
# SEARCH EXECUTION
# ==========================================

if search_btn:
    st.session_state.errors = []
    frames = []

    if not collections:
        st.warning("Please select at least one SPHEREx collection.")
    else:
        for col in collections:
            st.write(f"Searching: `{col}`")
            try:
                df = search_archive(col, ra, dec, radius)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                st.session_state.errors.append(f"{col}: {exc}")

        if frames:
            st.session_state.results = pd.concat(frames, ignore_index=True)
        else:
            st.session_state.results = pd.DataFrame()

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors"):
        for err in st.session_state.errors:
            st.error(err)

# ==========================================
# RESULTS & FITS VISUALIZATION
# ==========================================

results_df = st.session_state.results

if not results_df.empty:
    st.success(f"NASA archive returned {len(results_df)} observations.")

    st.subheader("🛰️ Archive Observations")

    wanted_cols = (
        "selected_collection", "obs_id", "s_ra", "s_dec",
        "t_min", "t_max", "em_min", "em_max", "access_url"
    )
    show_cols = [c for c in wanted_cols if c in results_df.columns]

    st.dataframe(results_df[show_cols] if show_cols else results_df, use_container_width=True)

    st.download_button(
        "📥 Download Archive CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="spherex_observations.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("🌌 SPHEREx Image Explorer")

    # डेटाफ्रेममधील कोणत्याही कॉलममध्ये FITS URL असल्यास ती शोधणे
    fits_urls = []
    for col in results_df.columns:
        urls = [
            str(u).strip() for u in results_df[col].dropna()
            if str(u).strip().startswith("http")
        ]
        if urls:
            fits_urls = urls
            break

    if fits_urls:
        c1, c2 = st.columns(2)
        with c1:
            idx1 = st.selectbox("Select Observation 1", range(len(fits_urls)), format_func=lambda i: f"Observation #{i+1}")
            show_image(fits_urls[idx1], f"Observation #{idx1+1}")
        if len(fits_urls) > 1:
            with c2:
                idx2 = st.selectbox("Select Observation 2", range(len(fits_urls)), index=1, format_func=lambda i: f"Observation #{i+1}")
                show_image(fits_urls[idx2], f"Observation #{idx2+1}")
    else:
        st.warning("No direct FITS URLs were returned. Archive metadata is still available.")
else:
    if search_btn and not st.session_state.errors:
        st.info("No observations found at these coordinates. Try increasing the Search Radius.")

# ==========================================
# ABOUT SECTION
# ==========================================

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
st.markdown("""
### Project Features
* NASA/IPAC-IRSA archive connectivity
* Astronomical coordinate search
• SPHEREx DR1 and QR3 datasets
• Observation metadata
• FITS image visualization
• Two-observation comparison
• CSV data export
• Automatic request retries
• Error handling
""")
st.caption("Data Source: NASA/IPAC Infrared Science Archive (IRSA)")
