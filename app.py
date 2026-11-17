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
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Note: Removed "Accept: text/csv" so VOTABLE XML returns cleanly
    session.headers.update({
        "User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"
    })

    return session

# ==========================================
# ARCHIVE SEARCH (VOTABLE - ZERO COLUMN SHIFT)
# ==========================================

@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = make_session()

    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}",
        "RESPONSEFORMAT": "VOTABLE"
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
        raise ValueError("NASA archive returned an HTML/proxy error.")

    # Check if IRSA returned a VOTable QUERY_STATUS ERROR
    if b'name="query_status" value="error"' in lower_head:
        raise ValueError("NASA IRSA reported a query parameter fault.")

    # Parse VOTable XML cleanly so commas in text never shift columns
    table = parse_single_table(io.BytesIO(raw)).to_table()
    df = table.to_pandas()

    # Decode any byte-string columns to normal UTF-8 strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

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
                return data.astype(np.float32), (hdu.name or "PRIMARY")

    raise ValueError("No valid 2D image found in FITS file.")

# ==========================================
# IMAGE DISPLAY
# ==========================================

def show_image(url, heading):
    st.markdown(f"### {heading}")

    try:
        with st.spinner("Loading infrared image from NASA IRSA..."):
            image, extension = load_fits(url)

        finite = image[np.isfinite(image)]
        if finite.size == 0:
            raise ValueError("FITS image contains no finite pixel values.")

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
# HEADER & SIDEBAR CONTROLS
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")

st.write(
    "Explore real infrared observations from "
    "NASA/IPAC Infrared Science Archive (IRSA)."
)

with st.sidebar:
    st.header("🔭 Target Coordinates")

    preset = st.selectbox(
        "Quick Sky Presets:",
        [
            "Custom Coordinates",
            "Orion Nebula (M42)",
            "Galactic Center",
            "Andromeda Galaxy (M31)",
            "Large Magellanic Cloud (LMC)"
        ]
    )

    preset_coords = {
        "Custom Coordinates": (127.8000, 40.6000),
        "Orion Nebula (M42)": (83.8221, -5.3911),
        "Galactic Center": (266.4168, -29.0078),
        "Andromeda Galaxy (M31)": (10.6847, 41.2687),
        "Large Magellanic Cloud (LMC)": (80.8942, -69.7561)
    }

    default_ra, default_dec = preset_coords[preset]

    ra = st.number_input(
        "Right Ascension (RA deg)",
        min_value=0.0,
        max_value=360.0,
        value=default_ra,
        format="%.4f"
    )

    dec = st.number_input(
        "Declination (Dec deg)",
        min_value=-90.0,
        max_value=90.0,
        value=default_dec,
        format="%.4f"
    )

    radius = st.number_input(
        "Search Radius (deg)",
        min_value=0.01,
        max_value=5.0,
        value=0.50,
        step=0.05,
        format="%.2f"
    )

    st.divider()
    st.header("🛰️ Collections")

    collections = st.multiselect(
        "Select SPHEREx Collections:",
        options=["spherex_dr1", "spherex_qr3", "spherex_dr1_deep", "spherex_dr1_cube"],
        default=["spherex_dr1", "spherex_qr3"]
    )

    search_clicked = st.button(
        "🚀 Search NASA Archive",
        type="primary",
        use_container_width=True
    )

# ==========================================
# SEARCH EXECUTION
# ==========================================

if search_clicked:
    st.session_state.searched = True
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []

    if not collections:
        st.warning("Please select at least one collection in the sidebar.")
    else:
        collected_frames = []

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

# ==========================================
# DISPLAY ERRORS
# ==========================================

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors", expanded=True):
        for err in st.session_state.errors:
            st.error(err)

# ==========================================
# DISPLAY ARCHIVE OBSERVATIONS
# ==========================================

if st.session_state.searched:
    results_df = st.session_state.results

    if not results_df.empty:
        st.success(f"NASA archive returned {len(results_df)} observations.")

        st.subheader("🛰️ Archive Observations")

        preferred_cols = [
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
        ]
        visible_cols = [c for c in preferred_cols if c in results_df.columns]

        st.dataframe(
            results_df[visible_cols] if visible_cols else results_df,
            use_container_width=True
        )

        # CSV Download Button
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Archive CSV",
            data=csv_bytes,
            file_name=f"spherex_ra{ra}_dec{dec}.csv",
            mime="text/csv"
        )

        # ==========================================
        # SPHEREx IMAGE EXPLORER
# ==========================================
st.divider()
st.subheader("🌌 SPHEREx Image Explorer")
fits_df = pd.DataFrame()
if "access_url" in results_df.columns:
fits_df = results_df[
results_df["access_url"]
.astype(str)
.str.contains(r"https?://.*.fits", case=False, na=False)
].reset_index(drop=True)
if not fits_df.empty:
selected_idx = st.selectbox(
f"Select an infrared FITS image to render ({len(fits_df)} available):",
options=range(len(fits_df)),
format_func=lambda i: (
f"[{fits_df.loc[i, 'selected_collection']}] "
f"{fits_df.loc[i].get('obs_id', f'Observation #{i+1}')} "
f"(RA: {fits_df.loc[i].get('s_ra', 'N/A')}, Dec: {fits_df.loc[i].get('s_dec', 'N/A')})"
)
)
chosen_row = fits_df.loc[selected_idx]
chosen_url = chosen_row["access_url"]
chosen_title = f"SPHEREx Observation: {chosen_row.get('obs_id', selected_idx + 1)}"
show_image(chosen_url, chosen_title)
else:
st.warning("No direct FITS URLs were returned. Archive metadata is still available.")
elif not st.session_state.errors:
st.info("No observations found in this sky region. Try another preset or increase the search radius.")

==========================================


ABOUT SECTION


==========================================

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write(
"MECH ORBIT is a student astronomy project designed to "
"explore public SPHEREx infrared observations."
)
