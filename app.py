import io
import urllib3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits
from astropy.io.votable import parse_single_table
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
        allowed_methods=["GET"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"
    })

    return session

# ==========================================
# ARCHIVE SEARCH
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

    raw = response.content.lstrip()
    lower_head = raw[:3000].lower()

    if b"<html" in lower_head or b"<!doctype" in lower_head or b"proxy error" in lower_head:
        raise ValueError("NASA archive returned an HTML page instead of CSV data.")

    if raw.startswith(b"<?xml") or b"<votable" in lower_head:
        table = parse_single_table(io.BytesIO(raw)).to_table()
        df = table.to_pandas()
        for col in df.select_dtypes(include=("object",)).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
    else:
        df = pd.read_csv(io.BytesIO(raw))

    df.columns = [str(c).strip().lower() for c in df.columns]
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
                data = data[data.shape[0] // 2]

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
        low, high = np.percentile(finite, [1, 99.5])

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
            f"Size: {image.shape[1]} × {image.shape[0]} px"
        )

    except Exception as e:
        st.error(f"Could not render FITS image: {e}")

# ==========================================
# UI HEADER & SIDEBAR CONTROLS
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive (IRSA).")

with st.sidebar:
    st.header("🔭 Search Parameters")

    collections = st.multiselect(
        "Select Datasets",
        options=["spherex_dr1", "spherex_qr3"],
        default=["spherex_dr1", "spherex_qr3"]
    )

    ra = st.number_input("Right Ascension (RA in deg)", min_value=0.0, max_value=360.0, value=127.6944, format="%.4f")
    dec = st.number_input("Declination (Dec in deg)", min_value=-90.0, max_value=90.0, value=-39.1776, format="%.4f")
    radius = st.number_input("Search Radius (deg)", min_value=0.01, max_value=5.0, value=0.10, step=0.05, format="%.2f")

    search_btn = st.button("Search NASA Archive", type="primary", use_container_width=True)

# ==========================================
# SEARCH EXECUTION
# ==========================================

if search_btn:
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []

    if not collections:
        st.warning("Please select at least one dataset from the sidebar.")
    else:
        all_dfs = []
        for col in collections:
            st.write(f"Searching: **{col}** ...")
            try:
                df_part = search_archive(col, ra, dec, radius)
                if not df_part.empty:
                    all_dfs.append(df_part)
            except Exception as err:
                st.session_state.errors.append(f"{col}: {err}")

        if all_dfs:
            st.session_state.results = pd.concat(all_dfs, ignore_index=True)

# ==========================================
# ERRORS & RESULTS DISPLAY
# ==========================================

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors", expanded=True):
        for err in st.session_state.errors:
            st.error(err)

results_df = st.session_state.results

if not results_df.empty:
    st.success(f"Found {len(results_df)} observations!")

    st.subheader("📊 Observation Metadata")
    st.dataframe(results_df, use_container_width=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Metadata CSV",
        data=csv_data,
        file_name=f"spherex_obs_ra{ra}_dec{dec}.csv",
        mime="text/csv"
    )

    # FITS URLs शोधणे
    url_col = None
    for candidate in ["access_url", "sia_url", "cloud_access", "url"]:
        if candidate in results_df.columns:
            url_col = candidate
            break

    if url_col:
        valid_rows = results_df[results_df[url_col].notna()].reset_index(drop=True)
        valid_urls = valid_rows[url_col].tolist()

        if valid_urls:
            st.divider()
            st.subheader("🖼️ FITS Image Visualization & Comparison")

            if len(valid_urls) == 1:
                show_image(valid_urls[0], "Observation #1")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    idx1 = st.selectbox(
                        "Select First Observation",
                        range(len(valid_urls)),
                        index=0,
                        format_func=lambda i: f"Obs #{i+1} ({valid_rows.iloc[i].get('selected_collection', '')})"
                    )
                    show_image(valid_urls[idx1], f"Observation #{idx1+1}")

                with col2:
                    idx2 = st.selectbox(
                        "Select Second Observation",
                        range(len(valid_urls)),
                        index=min(1, len(valid_urls) - 1),
                        format_func=lambda i: f"Obs #{i+1} ({valid_rows.iloc[i].get('selected_collection', '')})"
                    )
                    show_image(valid_urls[idx2], f"Observation #{idx2+1}")
else:
    if search_btn and not st.session_state.errors:
        st.info("No observations found for these coordinates. Try increasing the Search Radius or changing RA/Dec.")
    elif not search_btn:
        st.info("Enter coordinates in the sidebar and press Search NASA Archive.")

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
* SPHEREx DR1 and QR3 datasets
* Observation metadata
* FITS image visualization
* Two-observation comparison
* CSV data export
* Automatic request retries
* Error handling
""")
st.caption("Data Source: NASA/IPAC Infrared Science Archive (IRSA)")
