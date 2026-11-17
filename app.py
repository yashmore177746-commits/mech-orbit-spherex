
import io
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch

# ==========================================
# MECH ORBIT - SPHEREx SKY EXPLORER
# ==========================================

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

# ==========================================
# SESSION
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
        "User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0",
        "Accept": "text/csv"
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

    content = response.content.lstrip().lower()

    if content.startswith(b"<html") or b"proxy error" in content[:3000]:
        raise ValueError("NASA archive returned an HTML/proxy error.")

    df = pd.read_csv(io.BytesIO(response.content))

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    if not df.empty:
        if not any(c in df.columns for c in ["access_url", "s_ra"]):
            raise ValueError("Unexpected archive response format.")

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

    with fits.open(
        io.BytesIO(response.content),
        memmap=False
    ) as hdus:

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
# HEADER
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")

st.write(
    "Explore real infrared observations from "
    "NASA/IPAC Infrared Science Archive."
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("🔭 Search the Sky")

    ra = st.number_input(
        "Right Ascension (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=127.69444
    )

    dec = st.number_input(
        "Declination (degrees)",
        min_value=-90.0,
        max_value=90.0,
        value=-39.17760
    )

    radius = st.number_input(
        "Search Radius (degrees)",
        min_value=0.001,
        max_value=2.0,
        value=0.05,
        step=0.01
    )

    collections = st.multiselect(
        "SPHEREx Collections",
        [
            "spherex_dr1",
            "spherex_dr1_deep",
            "spherex_qr3",
            "spherex_qr3_deep"
        ],
        default=[
            "spherex_dr1",
            "spherex_qr3"
        ]
    )

    search = st.button(
        "🚀 Search NASA Archive",
        type="primary",
        use_container_width=True
    )

    if st.button("🔄 Clear Results", use_container_width=True):
        st.session_state.results = pd.DataFrame()
        st.session_state.errors = []
        st.rerun()

    st.caption("Source: NASA/IPAC-IRSA")


# ==========================================
# SEARCH EXECUTION
# ==========================================

if search:

    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []

    if not collections:

        st.warning("Please select at least one collection.")

    else:

        frames = []

        progress = st.progress(0)

        for i, collection in enumerate(collections):

            st.write(f"Searching: {collection}")

            try:

                frame = search_archive(
                    collection,
                    ra,
                    dec,
                    radius
                )

                if not frame.empty:

                    frame["selected_collection"] = collection
                    frames.append(frame)

            except Exception as error:

                st.session_state.errors.append(
                    f"{collection}: {str(error)}"
                )

            progress.progress((i + 1) / len(collections))

        progress.empty()

        if frames:

            st.session_state.results = pd.concat(
                frames,
                ignore_index=True
            )

# ==========================================
# RESULTS
# ==========================================

results = st.session_state.results

if st.session_state.errors:

    st.warning(
        "Some NASA archive requests could not be completed."
    )

    with st.expander("View search errors"):

        for error in st.session_state.errors:
            st.write(error)

if results.empty:

    st.info(
        "Enter coordinates and press Search NASA Archive."
    )

    if st.session_state.errors:
        st.error(
            "No archive results were received. "
            "The remote NASA service may be temporarily unavailable. "
            "Please retry later."
        )

else:

    st.success(
        f"NASA archive returned {len(results)} observations."
    )

    st.subheader("🛰️ Archive Observations")

    preferred = [
        "selected_collection",
        "obs_id",
        "s_ra",
        "s_dec",
        "t_min",
        "t_max",
        "em_min",
        "em_max",
        "access_url"
    ]

    display_columns = [
        c for c in preferred if c in results.columns
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # CSV DOWNLOAD

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Archive CSV",
        csv_data,
        file_name="mech_orbit_results.csv",
        mime="text/csv"
    )

    # ======================================
    # IMAGE URLS
    # ======================================

    st.divider()
    st.subheader("🌌 SPHEREx Image Explorer")

    if "access_url" in results.columns:

        usable = results.copy()

        usable["preview_url"] = usable["access_url"].apply(
            lambda x: str(x).strip()
            if pd.notna(x) and str(x).strip().startswith("https://")
            else None
        )

        usable = usable[
            usable["preview_url"].notna()
        ].reset_index(drop=True)

    else:
        usable = pd.DataFrame()

    if usable.empty:

        st.warning(
            "No direct FITS URLs were returned. "
            "Archive metadata is still available."
        )

    else:

        st.success(
            f"{len(usable)} observations have FITS URLs."
        )

        labels = []

        for i, row in usable.iterrows():

            labels.append(
                f"{i+1}. "
                f"{row.get('selected_collection', 'SPHEREx')} - "
                f"{row.get('obs_id', 'Observation')}"
            )

        st.subheader("🔬 Compare Two Observations")

        col1, col2 = st.columns(2)

        with col1:

            index_a = st.selectbox(
                "Observation A",
                range(len(usable)),
                format_func=lambda i: labels[i],
                key="obs_a"
            )

        with col2:

            index_b = st.selectbox(
                "Observation B",
                range(len(usable)),
                index=min(1, len(usable)-1),
                format_func=lambda i: labels[i],
                key="obs_b"
            )

        with col1:

            show_image(
                usable.iloc[index_a]["preview_url"],
                "Observation A"
            )

        with col2:

            show_image(
                usable.iloc[index_b]["preview_url"],
                "Observation B"
            )


# ==========================================
# ABOUT
# ==========================================

st.divider()

st.header("🚀 About MECH ORBIT")

st.write(
    "MECH ORBIT is a student astronomy project "
    "designed to explore public SPHEREx infrared observations."
)

st.subheader("Project Features")

st.markdown("""
- NASA/IPAC-IRSA archive connectivity
- Astronomical coordinate search
- SPHEREx DR1 and QR3 datasets
- Observation metadata
- FITS image visualization
- Two-observation comparison
- CSV data export
- Automatic request retries
- Error handling
""")

st.subheader("Data Source")

st.write("NASA/IPAC Infrared Science Archive (IRSA)")

st.subheader("Team")

st.write("MECH ORBIT — Student Project")

st.caption("MECH ORBIT | SPHEREx Sky Explorer")
