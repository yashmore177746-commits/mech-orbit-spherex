import io
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

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MECH-ORBIT/1.0"
    })

    return session

# ==========================================
# ARCHIVE SEARCH (VOTable Support)
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = make_session()

    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}"
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

    table = parse_single_table(io.BytesIO(response.content)).to_table()
    df = table.to_pandas()

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    df.columns = [str(c).strip().lower() for c in df.columns]
    df["collection"] = collection

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
            f"Size: {image.shape[1]} × {image.shape[0]} pixels"
        )

        st.link_button(
            "Open Original FITS",
            url,
            use_container_width=True
        )

    except Exception as error:
        st.warning(f"Image unavailable: {error}")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================

with st.sidebar:
    st.header("🔭 Search the Sky")

    ra = st.number_input(
        "Right Ascension (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=127.69,
        step=0.1,
        format="%.4f"
    )

    dec = st.number_input(
        "Declination (degrees)",
        min_value=-90.0,
        max_value=90.0,
        value=-39.17,
        step=0.1,
        format="%.4f"
    )

    radius = st.number_input(
        "Search Radius (degrees)",
        min_value=0.01,
        max_value=5.0,
        value=0.10,
        step=0.05,
        format="%.2f"
    )

    collections = st.multiselect(
        "SPHEREx Collections",
        options=[
            "spherex_dr1",
            "spherex_qr3",
            "spherex_dr1_deep",
            "spherex_qr3_deep"
        ],
        default=["spherex_dr1", "spherex_qr3"]
    )

    search_btn = st.button("🚀 Search NASA Archive", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Results", use_container_width=True)

    st.caption("Source: NASA/IPAC IRSA")

if clear_btn:
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []
    st.rerun()

# ==========================================
# MAIN UI & SEARCH EXECUTION
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive.")

if search_btn:
    st.session_state.errors = []
    all_dfs = []

    if not collections:
        st.warning("Please select at least one SPHEREx collection.")
    else:
        for col in collections:
            st.write(f"Searching: `{col}`...")
            try:
                df = search_archive(col, ra, dec, radius)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                st.session_state.errors.append(f"{col}: {str(e)}")

        if all_dfs:
            st.session_state.results = pd.concat(all_dfs, ignore_index=True)
        else:
            st.session_state.results = pd.DataFrame()

# ==========================================
# DISPLAY ERRORS & RESULTS
# ==========================================

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors"):
        for err in st.session_state.errors:
            st.error(err)

results_df = st.session_state.results

if not results_df.empty:
    st.success(f"Found {len(results_df)} observations!")

    display_cols = [
        c for c in ["collection", "obs_id", "s_ra", "s_dec", "em_min", "em_max", "t_exptime", "access_url"]
        if c in results_df.columns
    ]

    st.dataframe(results_df[display_cols] if display_cols else results_df, use_container_width=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"spherex_ra{ra}_dec{dec}.csv",
        mime="text/csv"
    )

    # FITS Image Comparison
    if "access_url" in results_df.columns:
        valid_urls = results_df["access_url"].dropna().tolist()
        if valid_urls:
            st.divider()
            st.subheader("🖼️ FITS Image Visualization & Comparison")

            col1, col2 = st.columns(2)
            with col1:
                idx1 = st.selectbox("Select Observation 1", range(len(valid_urls)), format_func=lambda i: f"Obs #{i+1} ({results_df.iloc[i].get('collection', '')})")
                show_image(valid_urls[idx1], f"Observation #{idx1+1}")

            if len(valid_urls) > 1:
                with col2:
                    idx2 = st.selectbox("Select Observation 2", range(len(valid_urls)), index=1, format_func=lambda i: f"Obs #{i+1} ({results_df.iloc[i].get('collection', '')})")
                    show_image(valid_urls[idx2], f"Observation #{idx2+1}")
else:
    if search_btn and not st.session_state.errors:
        st.info("No observations found for these coordinates. Try increasing the Search Radius or changing RA/Dec.")
    elif not search_btn:
        st.info("Enter coordinates and press Search NASA Archive.")

# ==========================================
# ABOUT SECTION
# ==========================================

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
st.caption("Data Source: NASA/IPAC Infrared Science Archive (IRSA) | Team: MECH ORBIT — Student Project")
