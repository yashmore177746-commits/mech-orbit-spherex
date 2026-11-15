
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    AsinhStretch
)

# ==========================================
# MECH ORBIT - SPHEREx SKY EXPLORER
# ==========================================

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Sky Explorer",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 MECH ORBIT — SPHEREx Sky Explorer")

st.markdown(
    """
    Explore real public SPHEREx infrared observations
    using NASA/IPAC-IRSA astronomical archive data.
    """
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
        options=[
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
        "🚀 Search SPHEREx Images",
        type="primary",
        use_container_width=True
    )

    st.caption(
        "Data source: NASA/IPAC Infrared Science Archive"
    )


# ==========================================
# SIA2 ARCHIVE SEARCH
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def query_archive(collection, ra_value, dec_value, radius_value):

    params = {
        "COLLECTION": collection,
        "POS": (
            f"circle {ra_value} "
            f"{dec_value} {radius_value}"
        )
    }

    response = requests.get(
        SIA_URL,
        params=params,
        headers={
            "Accept": "application/x-votable+xml"
        },
        timeout=120
    )

    response.raise_for_status()

    if not response.content.strip():
        raise ValueError("Empty response received from NASA archive.")

    # Read official SIA2 VOTable response
    table = Table.read(
        io.BytesIO(response.content),
        format="votable"
    )

    df = table.to_pandas()

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    return df


# ==========================================
# FIND DIRECT FITS URL
# ==========================================

def get_access_url(row):

    for column in [
        "access_url",
        "accessurl"
    ]:

        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if value.startswith(("https://", "http://")):
            return value

    return None


# ==========================================
# READ FITS IMAGE
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def read_fits_image(url):

    response = requests.get(
        url,
        timeout=180,
        headers={"User-Agent": "MECH-ORBIT-SPHEREx-Explorer"}
    )

    response.raise_for_status()

    with fits.open(
        io.BytesIO(response.content),
        memmap=False
    ) as hdus:

        image = None
        image_name = "Image"

        # Search all FITS extensions
        for hdu in hdus:

            data = getattr(hdu, "data", None)

            if data is None:
                continue

            data = np.asarray(data)

            if data.ndim < 2:
                continue

            # Select middle plane for 3D or higher data
            while data.ndim > 2:
                data = data[data.shape[0] // 2]

            if data.ndim == 2:

                data = data.astype(np.float32)

                if np.isfinite(data).any():

                    image = data
                    image_name = str(hdu.name)
                    break

        if image is None:

            raise ValueError(
                "No valid 2D image found in FITS extensions."
            )

        return image, image_name


# ==========================================
# SESSION STATE
# ==========================================

if "archive_results" not in st.session_state:
    st.session_state.archive_results = pd.DataFrame()

if "search_errors" not in st.session_state:
    st.session_state.search_errors = []


# ==========================================
# SEARCH
# ==========================================

if search:

    st.session_state.archive_results = pd.DataFrame()
    st.session_state.search_errors = []

    if not collections:

        st.warning("Please select at least one collection.")

    else:

        frames = []

        progress = st.progress(0)

        with st.spinner(
            "Connecting to NASA/IPAC-IRSA archive..."
        ):

            for i, collection in enumerate(collections):

                try:

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

                progress.progress(
                    (i + 1) / len(collections)
                )

        if frames:

            st.session_state.archive_results = pd.concat(
                frames,
                ignore_index=True
            )

        progress.empty()


# ==========================================
# DISPLAY ERRORS
# ==========================================

for error in st.session_state.search_errors:

    st.error(error)


# ==========================================
# DISPLAY ARCHIVE RESULTS
# ==========================================

results = st.session_state.archive_results

if results.empty:

    st.info(
        "Enter sky coordinates in the sidebar "
        "and click Search SPHEREx Images."
    )

else:

    st.success(
        f"NASA Archive returned {len(results)} observations!"
    )

    st.subheader("🛰️ Archive Observations")

    # Display real archive metadata
    preferred_columns = [
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
        c for c in preferred_columns
        if c in results.columns
    ]

    if display_columns:

        st.dataframe(
            results[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    # Download archive metadata
    csv_data = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Archive Results CSV",
        data=csv_data,
        file_name="mech_orbit_spherex_results.csv",
        mime="text/csv"
    )

    # ======================================
    # FITS URL SELECTION
    # ======================================

    results = results.copy()

    results["preview_url"] = results.apply(
        get_access_url,
        axis=1
    )

    usable = results[
        results["preview_url"].notna()
    ].reset_index(drop=True)

    st.divider()

    st.subheader("🌌 SPHEREx Image Explorer")

    if usable.empty:

        st.warning(
            "Archive search succeeded, but no direct "
            "FITS access_url was returned."
        )

        st.write(
            "Check the archive response metadata. "
            "Image preview requires a valid FITS URL."
        )

    else:

        st.success(
            f"{len(usable)} observations have direct FITS URLs."
        )

        # ==================================
        # OBSERVATION LABELS
        # ==================================

        labels = []

        for i, row in usable.iterrows():

            collection = row.get(
                "selected_collection",
                "SPHEREx"
            )

            obs_id = row.get(
                "obs_id",
                f"Observation {i + 1}"
            )

            labels.append(
                f"{i + 1}. {collection} — {obs_id}"
            )

        # ==================================
        # COMPARE OBSERVATIONS
        # ==================================

        st.subheader("🔬 Compare Two Observations")

        col1, col2 = st.columns(2)

        with col1:

            index_a = st.selectbox(
                "Select Observation A",
                options=range(len(usable)),
                format_func=lambda i: labels[i],
                key="observation_a"
            )

        with col2:

            index_b = st.selectbox(
                "Select Observation B",
                options=range(len(usable)),
                index=min(1, len(usable) - 1),
                format_func=lambda i: labels[i],
                key="observation_b"
            )

        # ==================================
        # IMAGE DISPLAY FUNCTION
        # ==================================

        def display_observation(column, index, heading):

            with column:

                st.markdown(f"### {heading}")

                st.write(labels[index])

                image_url = usable.iloc[index]["preview_url"]

                try:

                    with st.spinner(
                        "Downloading FITS image..."
                    ):

                        image, extension_name = read_fits_image(
                            image_url
                        )

                    st.caption(
                        f"FITS extension: {extension_name} | "
                        f"Image size: {image.shape[1]} × {image.shape[0]} pixels"
                    )

                    fig, ax = plt.subplots(
                        figsize=(6, 5)
                    )

                    norm = ImageNormalize(
                        image,
                        interval=PercentileInterval(99.5),
                        stretch=AsinhStretch()
                    )

                    ax.imshow(
                        image,
                        origin="lower",
                        cmap="magma",
                        norm=norm
                    )

                    ax.set_xlabel("Pixel X")
                    ax.set_ylabel("Pixel Y")
                    ax.set_title(heading)

                    st.pyplot(
                        fig,
                        clear_figure=True
                    )

                    plt.close(fig)

                except Exception as error:

                    st.error(
                        f"FITS image could not be displayed: {error}"
                    )

                st.link_button(
                    "🔗 Open Original FITS File",
                    image_url,
                    use_container_width=True
                )

        # ==================================
        # DISPLAY BOTH IMAGES
        # ==================================

        display_observation(
            col1,
            index_a,
            "Observation A"
        )

        display_observation(
            col2,
            index_b,
            "Observation B"
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

st.divider()

st.subheader("🚀 About MECH ORBIT")

st.markdown(
    """
    **MECH ORBIT** is a student astronomy project
    designed to explore public SPHEREx infrared observations.

    ### Project Features

    - NASA/IPAC-IRSA archive connectivity
    - Astronomical coordinate-based search
    - SPHEREx observation metadata
    - FITS image downloading
    - Infrared image visualization
    - Two-observation comparison
    - Archive results CSV download

    ### Data Source

    NASA/IPAC Infrared Science Archive (IRSA)

    ### Team

    **MECH ORBIT — Student Project**
    """
)

st.caption(
    "MECH ORBIT | SPHEREx Sky Explorer | NASA/IPAC-IRSA"
)
