import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

# ==========================================
# MECH ORBIT - SPHEREx SKY EXPLORER
# ==========================================

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.caption("Explore real infrared observations from NASA/IPAC-IRSA.")

# ==========================================
# CUSTOM STYLE
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #080d1b;
}
[data-testid="stSidebar"] {
    background-color: #111a2e;
}
h1, h2, h3 {
    color: #8bd5ff;
}
.stButton button {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🔭 Search the Sky")

    ra = st.number_input(
        "Right Ascension (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=127.69444,
        step=0.1
    )

    dec = st.number_input(
        "Declination (degrees)",
        min_value=-90.0,
        max_value=90.0,
        value=-39.17760,
        step=0.1
    )

    radius = st.number_input(
        "Search Radius (degrees)",
        min_value=0.001,
        max_value=2.0,
        value=0.05,
        step=0.01,
        format="%.3f"
    )

    collections = st.multiselect(
        "SPHEREx Collections",
        [
            "spherex_dr1",
            "spherex_dr1_deep",
            "spherex_dr1_cube",
            "spherex_qr3",
            "spherex_qr3_deep"
        ],
        default=["spherex_dr1", "spherex_qr3"]
    )

    search = st.button(
        "🚀 Search NASA Archive",
        type="primary",
        use_container_width=True
    )

    st.divider()
    st.caption("Official source: NASA/IPAC-IRSA")

# ==========================================
# ARCHIVE QUERY
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def query_archive(collection, ra_value, dec_value, radius_value):

    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra_value} {dec_value} {radius_value}",
        "RESPONSEFORMAT": "VOTABLE"
    }

    response = requests.get(
        SIA_URL,
        params=params,
        headers={"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"},
        timeout=120
    )

    response.raise_for_status()

    if not response.content.strip():
        raise ValueError("Empty response from NASA archive.")

    table = Table.read(
        io.BytesIO(response.content),
        format="votable"
    )

    df = table.to_pandas()

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    return df

# ==========================================
# FIND FITS URL
# ==========================================

def get_access_url(row):

    possible_columns = [
        "access_url",
        "accessurl"
    ]

    for column in possible_columns:

        if column not in row.index:
            continue

        value = row[column]

        if value is None:
            continue

        if isinstance(value, str):

            value = value.strip()

            if value.startswith(("https://", "http://")):
                return value

    return None

# ==========================================
# FITS IMAGE READER
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def read_fits_image(url):

    response = requests.get(
        url,
        headers={"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"},
        timeout=180
    )

    response.raise_for_status()

    if not response.content.startswith(b"SIMPLE"):
        # FITS files may also contain other valid FITS extensions.
        # Astropy performs the definitive format validation below.
        pass

    with fits.open(
        io.BytesIO(response.content),
        memmap=False,
        ignore_missing_simple=True
    ) as hdus:

        candidates = []

        for hdu in hdus:

            data = getattr(hdu, "data", None)

            if data is None:
                continue

            data = np.asarray(data)

            if data.ndim < 2:
                continue

            while data.ndim > 2:
                data = data[data.shape[0] // 2]

            if data.ndim == 2:

                data = data.astype(np.float32)

                if np.isfinite(data).any():

                    candidates.append((
                        data,
                        str(hdu.name)
                    ))

        if not candidates:
            raise ValueError(
                "No valid two-dimensional image found in FITS file."
            )

        # Select the largest valid image plane.
        image, extension = max(
            candidates,
            key=lambda item: item[0].size
        )

        return image, extension

# ==========================================
# IMAGE VISUALIZATION
# ==========================================

def show_image(image, title):

    finite = image[np.isfinite(image)]

    if finite.size == 0:
        st.error("Image contains no valid pixels.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    norm = ImageNormalize(
        finite,
        interval=PercentileInterval(99.5),
        stretch=AsinhStretch()
    )

    ax.imshow(
        np.ma.masked_invalid(image),
        origin="lower",
        cmap="magma",
        norm=norm
    )

    ax.set_title(title, color="white")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="white")

    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

# ==========================================
# SESSION STATE
# ==========================================

if "archive_results" not in st.session_state:
    st.session_state.archive_results = pd.DataFrame()

if "search_errors" not in st.session_state:
    st.session_state.search_errors = []

# ==========================================
# SEARCH NASA
# ==========================================

if search:

    st.session_state.archive_results = pd.DataFrame()
    st.session_state.search_errors = []

    if not collections:

        st.warning("Please select a SPHEREx collection.")

    else:

        frames = []

        progress = st.progress(0)

        for i, collection in enumerate(collections):

            try:

                with st.spinner(f"Searching {collection}..."):

                    frame = query_archive(
                        collection,
                        ra,
                        dec,
                        radius
                    )

                if not frame.empty:

                    frame["selected_collection"] = collection
                    frames.append(frame)

            except Exception as error:

                st.session_state.search_errors.append(
                    f"{collection}: {error}"
                )

            progress.progress((i + 1) / len(collections))

        if frames:

            st.session_state.archive_results = pd.concat(
                frames,
                ignore_index=True
            )

        progress.empty()

# ==========================================
# SHOW ERRORS
# ==========================================

for error in st.session_state.search_errors:
    st.error(error)

# ==========================================
# ARCHIVE RESULTS
# ==========================================

results = st.session_state.archive_results

if results.empty:

    st.info(
        "🌌 Enter coordinates and press Search NASA Archive."
    )

else:

    st.success(
        f"NASA Archive returned {len(results)} observations."
    )

    st.subheader("🛰️ Archive Observations")

    preferred = [
        "selected_collection",
        "obs_id",
        "obs_publisher_did",
        "s_ra",
        "s_dec",
        "t_min",
        "t_max",
        "em_min",
        "em_max",
        "access_format",
        "access_url",
        "cloud_access"
    ]

    available = [
        col for col in preferred
        if col in results.columns
    ]

    if available:
        st.dataframe(
            results[available],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("Expected metadata columns were not returned.")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    # ======================================
    # DOWNLOAD CSV
    # ======================================

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download NASA Archive CSV",
        data=csv,
        file_name="mech_orbit_spherex_data.csv",
        mime="text/csv"
    )

    # ======================================
    # DEBUG DATA
    # ======================================

    with st.expander("🔍 Archive Debug Information"):

        st.write("Available archive columns:")
        st.write(results.columns.tolist())

        st.write("First archive record:")
        st.json(results.iloc[0].to_dict())

    # ======================================
    # IMAGE EXPLORER
    # ======================================

    st.divider()
    st.header("🌌 SPHEREx Image Explorer")

    results = results.copy()

    results["preview_url"] = results.apply(
        get_access_url,
        axis=1
    )

    usable = results[
        results["preview_url"].notna()
    ].reset_index(drop=True)

    if usable.empty:

        st.warning(
            "No direct access_url was found in these observations."
        )

        st.info(
            "Check Archive Debug Information for access_url "
            "and cloud_access fields. Cloud access may require "
            "a separate URL resolution step."
        )

    else:

        st.success(
            f"{len(usable)} observations have direct image URLs."
        )

        labels = []

        for i, row in usable.iterrows():

            obs_id = row.get(
                "obs_id",
                f"Observation {i + 1}"
            )

            collection = row.get(
                "selected_collection",
                "SPHEREx"
            )

            labels.append(
                f"{i + 1}. {collection} — {obs_id}"
            )

        # ==================================
        # SELECT TWO IMAGES
        # ==================================

        st.subheader("🔬 Compare Two Observations")

        col1, col2 = st.columns(2)

        with col1:

            index_a = st.selectbox(
                "Observation A",
                options=range(len(usable)),
                format_func=lambda i: labels[i],
                key="observation_a"
            )

        with col2:

            index_b = st.selectbox(
                "Observation B",
                options=range(len(usable)),
                index=min(1, len(usable) - 1),
                format_func=lambda i: labels[i],
                key="observation_b"
            )

        # ==================================
        # DISPLAY IMAGE
        # ==================================

        def display_observation(column, index, heading):

            with column:

                st.subheader(heading)

                st.caption(labels[index])

                image_url = usable.iloc[index]["preview_url"]

                try:

                    with st.spinner("Downloading FITS image..."):

                        image, extension = read_fits_image(image_url)

                    st.success("FITS image loaded successfully.")

                    st.caption(
                        f"Extension: {extension} | "
                        f"Dimensions: {image.shape[1]} × {image.shape[0]}"
                    )

                    show_image(image, heading)

                except Exception as error:

                    st.error(f"Image error: {error}")

                st.link_button(
                    "🔗 Open Original FITS",
                    image_url,
                    use_container_width=True
                )

        display_observation(col1, index_a, "Observation A")
        display_observation(col2, index_b, "Observation B")

# ==========================================
# ABOUT
# ==========================================

st.divider()

st.header("🚀 About MECH ORBIT")

st.markdown("""
MECH ORBIT is a student astronomy project for exploring
real SPHEREx infrared observations.

### Project Features

- NASA/IPAC-IRSA archive connectivity
- Right Ascension and Declination search
- SPHEREx DR1 and QR3 datasets
- Observation metadata
- FITS image downloading
- Infrared image visualization
- Two-observation comparison
- CSV data export
- Archive debugging information

### Data Source

NASA/IPAC Infrared Science Archive (IRSA)

### Team

**MECH ORBIT — Student Project**
""")

st.caption("MECH ORBIT | SPHEREx Sky Explorer")
