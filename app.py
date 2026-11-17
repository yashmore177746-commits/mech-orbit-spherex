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

SIA_URL = "https://caltech.edu"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = list()

if "searched" not in st.session_state:
    st.session_state.searched = False


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
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"})
    return session


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = make_session()
    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}",
        "RESPONSEFORMAT": "VOTABLE"
    }
    response = session.get(SIA_URL, params=params, timeout=(20, 120))
    response.raise_for_status()

    if not response.content:
        raise ValueError("Empty response from NASA archive.")

    raw = response.content.lstrip()
    lower_head = raw[:3000].lower()

    if b"<html" in lower_head or b"<!doctype" in lower_head or b"proxy error" in lower_head:
        raise ValueError("NASA archive returned an HTML/proxy error.")

    if b'name="query_status" value="error"' in lower_head:
        raise ValueError("NASA IRSA reported a query parameter fault.")

    table = parse_single_table(io.BytesIO(raw)).to_table()
    df = table.to_pandas()

    for col in df.select_dtypes(include=("object",)).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    df.columns = [str(c).strip().lower() for c in df.columns]
    df["selected_collection"] = collection
    return df


@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    session = make_session()
    response = session.get(url, timeout=(20, 180))
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
                mid = data.shape[0] // 2
                data = data[mid]
            if data.ndim == 2 and np.isfinite(data).any():
                return data.astype(np.float32), (hdu.name or "PRIMARY")

    raise ValueError("No valid 2D image found in FITS file.")


def show_image(url, heading):
    st.markdown(f"### {heading}")
    try:
        with st.spinner("Loading infrared image from NASA IRSA..."):
            image, extension = load_fits(url)

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError("FITS image contains no finite pixel values.")

        low, high = np.percentile(finite, (1, 99.5))
        if low >= high:
            low = float(finite.min())
            high = float(finite.max()) + 1e-6

        fig, ax = plt.subplots(figsize=(6, 5))
        norm = ImageNormalize(vmin=low, vmax=high, stretch=AsinhStretch())
        ax.imshow(np.ma.masked_invalid(image), origin="lower", cmap="magma", norm=norm)
        ax.set_title(heading)
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        height, width = image.shape
        st.caption(f"Extension: {extension} | Size: {width} × {height} pixels")
        st.link_button("Open Original FITS", url, use_container_width=True)

    except Exception as error:
        st.warning(f"Image unavailable: {error}")


def render_image_explorer(results_df):
    st.divider()
    st.subheader("🌌 SPHEREx Image Explorer")

    if "access_url" not in results_df.columns:
        st.warning("No access_url column returned by NASA archive.")
        return

    url_series = results_df["access_url"].astype(str)
    mask = url_series.str.contains(r"https?://.*\.fits", case=False, na=False)
    fits_df = results_df[mask].reset_index(drop=True)

    if fits_df.empty:
        st.warning("No direct FITS URLs were returned. Archive metadata is still available.")
        return

    labels = list()
    for i in range(len(fits_df)):
        col_name = fits_df.loc[i, "selected_collection"]
        obs_name = fits_df.loc[i].get("obs_id", f"Observation #{i+1}")
        ra_val = fits_df.loc[i].get("s_ra", "N/A")
        dec_val = fits_df.loc[i].get("s_dec", "N/A")
        labels.append(f"[{col_name}] {obs_name} (RA: {ra_val}, Dec: {dec_val})")

    selected_idx = st.selectbox(
        f"Select an infrared FITS image to render ({len(fits_df)} available):",
        options=list(range(len(fits_df))),
        format_func=lambda i: labels[i]
    )

    chosen_row = fits_df.loc[selected_idx]
    chosen_url = chosen_row["access_url"]
    chosen_title = f"SPHEREx Observation: {chosen_row.get('obs_id', selected_idx + 1)}"
    show_image(chosen_url, chosen_title)


# ==========================================
# MAIN APP UI
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive (IRSA).")

st.sidebar.header("🔭 Target Coordinates")

preset_options = (
    "Custom Coordinates",
    "Orion Nebula (M42)",
    "Galactic Center",
    "Andromeda Galaxy (M31)",
    "Large Magellanic Cloud (LMC)"
)

preset = st.sidebar.selectbox("Quick Sky Presets:", preset_options)

preset_coords = {
    "Custom Coordinates": (127.8000, 40.6000),
    "Orion Nebula (M42)": (83.8221, -5.3911),
    "Galactic Center": (266.4168, -29.0078),
    "Andromeda Galaxy (M31)": (10.6847, 41.2687),
    "Large Magellanic Cloud (LMC)": (80.8942, -69.7561)
}

default_ra, default_dec = preset_coords[preset]

ra = st.sidebar.number_input("Right Ascension (RA deg)", min_value=0.0, max_value=360.0, value=default_ra, format="%.4f")
dec = st.sidebar.number_input("Declination (Dec deg)", min_value=-90.0, max_value=90.0, value=default_dec, format="%.4f")
radius = st.sidebar.number_input("Search Radius (deg)", min_value=0.01, max_value=5.0, value=0.50, step=0.05, format="%.2f")

st.sidebar.divider()
st.sidebar.header("🛰️ Collections")

collection_options = ("spherex_dr1", "spherex_qr3", "spherex_dr1_deep", "spherex_dr1_cube")
default_collections = ("spherex_dr1", "spherex_qr3")

collections = st.sidebar.multiselect(
    "Select SPHEREx Collections:",
    options=collection_options,
    default=default_collections
)

search_clicked = st.sidebar.button("🚀 Search NASA Archive", type="primary", use_container_width=True)

if search_clicked:
    st.session_state.searched = True
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = list()

    if not collections:
        st.warning("Please select at least one collection in the sidebar.")
    else:
        collected_frames = list()
        for col in collections:
            st.write(f"Searching: **{col}** ...")
            try:
                df_part = search_archive(col, ra, dec, radius)
                if not df_part.empty:
                    collected_frames.append(df_part)
            except Exception as err:
                st.session_state.errors.append(f"{col}: {err}")

        if collected_frames:
            st.session_state.results = pd.concat(collected_frames, ignore_index=True)

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors", expanded=True):
        for err in st.session_state.errors:
            st.error(err)

if st.session_state.searched:
    results_df = st.session_state.results

    if not results_df.empty:
        st.success(f"NASA archive returned {len(results_df)} observations.")
        st.subheader("🛰️ Archive Observations")

        preferred_cols = (
            "selected_collection",
            "obs_id",
            "s_ra",
            "s_dec",
            "t_exptime",
            "em_min",
            "em_max",
            "dataproduct_type",
            "access_format",
            "access_url"
        )
        visible_cols = [c for c in preferred_cols if c in results_df.columns]
        st.dataframe(results_df[visible_cols] if visible_cols else results_df, use_container_width=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Archive CSV",
            data=csv_bytes,
            file_name=f"spherex_ra{ra}_dec{dec}.csv",
            mime="text/csv"
        )

        render_image_explorer(results_df)

    elif not st.session_state.errors:
        st.info("No observations found in this sky region. Try another preset or increase the search radius.")

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
