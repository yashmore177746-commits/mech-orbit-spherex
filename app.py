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

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# MECH ORBIT - SPHEREx SKY EXPLORER
# ==========================================

# FIXED: Official NASA/IPAC IRSA SIA2 API Endpoint
SIA_URL = "https://caltech.edu"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = []

if "searched" not in st.session_state:
    st.session_state.searched = False

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
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=["GET"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"
    })

    return session

# ==========================================
# ARCHIVE SEARCH (CACHED)
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
# FITS IMAGE DOWNLOAD & PARSING
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
                return data.astype(np.float32), (hdu.name or "PRIMARY")

    raise ValueError("No valid 2D image found in FITS file.")

# ==========================================
# IMAGE DISPLAY HELPER
# ==========================================

def show_image(url, heading):
    st.markdown(f"### {heading}")

    try:
        with st.spinner("Loading infrared FITS image from NASA IRSA..."):
            image, extension = load_fits(url)

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError("FITS image contains only NaN/Inf values.")

        low, high = np.percentile(finite, [1, 99.5])

        if low >= high:
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
# SIDEBAR CONTROLS
# ==========================================

with st.sidebar:
    st.header("🔭 Sky Coordinates")
    
    # Default coordinates (e.g., Galactic Center / Orion region)
    ra = st.number_input("Right Ascension (RA in deg)", min_value=0.0, max_value=360.0, value=83.8221, format="%.4f")
    dec = st.number_input("Declination (Dec in deg)", min_value=-90.0, max_value=90.0, value=-5.3911, format="%.4f")
    radius = st.number_input("Search Radius (deg)", min_value=0.01, max_value=5.0, value=0.5, step=0.1, format="%.2f")

    st.divider()
    st.header("🛰️ SPHEREx Collections")
    collections = st.multiselect(
        "Select collections to query:",
        options=["spherex_dr1", "spherex_qr3", "spherex_dr1_deep", "spherex_dr1_cube"],
        default=["spherex_dr1", "spherex_qr3"]
    )

    search_btn = st.button("🚀 Search NASA IRSA Archive", type="primary", use_container_width=True)

# ==========================================
# MAIN UI & SEARCH EXECUTION
# ==========================================

st.title("🌌 MECH ORBIT | SPHEREx Sky Explorer")
st.caption("Connected to NASA/IPAC Infrared Science Archive (IRSA).")

if search_btn:
    st.session_state.searched = True
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []

    if not collections:
        st.warning("Please select at least one SPHEREx collection from the sidebar.")
    else:
        all_dfs = []
        for col_name in collections:
            with st.spinner(f"Searching: {col_name} ..."):
                try:
                    df_part = search_archive(col_name, ra, dec, radius)
                    if not df_part.empty:
                        all_dfs.append(df_part)
                except Exception as exc:
                    st.session_state.errors.append(f"{col_name}: {exc}")

        if all_dfs:
            st.session_state.results = pd.concat(all_dfs, ignore_index=True)

# Display Warnings / Errors cleanly
if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors", expanded=True):
        for err in st.session_state.errors:
            st.error(err)

# Display Results & FITS Viewer
if st.session_state.searched:
    df_res = st.session_state.results

    if not df_res.empty:
        st.success(f"Found {len(df_res)} infrared observations!")

        # Filter out metadata/auxiliary rows if dataproduct_type exists
        if "access_url" in df_res.columns:
            valid_urls = df_res[df_res["access_url"].notna()].reset_index(drop=True)
        else:
            valid_urls = pd.DataFrame()

        # Show clean summary table
        display_cols = [c for c in ["selected_collection", "obs_id", "s_ra", "s_dec", "em_min", "em_max", "dataproduct_type", "access_format"] if c in df_res.columns]
        st.dataframe(df_res[display_cols] if display_cols else df_res, use_container_width=True)

        # FITS Image Viewer Section
        if not valid_urls.empty:
            st.divider()
            st.subheader("🖼️ Infrared FITS Image Viewer")

            # Filter specifically for FITS files if access_format is available
            if "access_format" in valid_urls.columns:
                fits_rows = valid_urls[valid_urls["access_format"].str.contains("fits", case=False, na=False)].reset_index(drop=True)
                if fits_rows.empty:
                    fits_rows = valid_urls
            else:
                fits_rows = valid_urls

            selected_idx = st.selectbox(
                "Select an observation to render:",
                options=range(len(fits_rows)),
                format_func=lambda i: f"[{fits_rows.loc[i, 'selected_collection']}] {fits_rows.loc[i, 'obs_id'] if 'obs_id' in fits_rows.columns else f'Observation #{i+1}'}"
            )

            selected_row = fits_rows.loc[selected_idx]
            obs_title = f"SPHEREx Observation: {selected_row.get('obs_id', selected_idx + 1)}"
            show_image(selected_row["access_url"], obs_title)

    elif not st.session_state.errors:
        st.info("No SPHEREx observations found at these coordinates yet. Try increasing the search radius or changing RA/Dec.")

# ==========================================
# ABOUT SECTION (FROM YOUR SCREENSHOT)
# ==========================================

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write(
    "MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations."
)

st.subheader("Project Features")
st.markdown("""
