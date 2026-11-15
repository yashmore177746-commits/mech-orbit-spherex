import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Sky Explorer",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 MECH ORBIT — SPHEREx Sky Explorer")

st.caption(
    "Explore NASA SPHEREx infrared observations "
    "using public NASA/IPAC-IRSA archive data."
)

# ---------------------------------
# SIDEBAR SEARCH
# ---------------------------------

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
    ["spherex_qr2", "spherex_qr2_deep"],
    default=["spherex_qr2"]
)
    

    search = st.button(
        "🚀 Search SPHEREx Images",
        type="primary",
        use_container_width=True
    )

# ---------------------------------
# ARCHIVE QUERY
# --------------------------------

@st.cache_data(ttl=1800)
def query_archive(collection, ra_value, dec_value, radius_value):

    from astropy.table import Table

    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra_value} {dec_value} {radius_value}",
        "FORMAT": "image/fits",
        "RESPONSEFORMAT": "VOTABLE"
    }

    response = requests.get(
        SIA_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

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


# ---------------------------------
# FIND FITS URL
# ---------------------------------

def get_access_url(row):

    for column in ["access_url", "accessurl"]:

        if column in row.index:

            value = row[column]

            if pd.notna(value):

                value = str(value).strip()

                if value.startswith("https://") or value.startswith("http://"):
                    return value

    return None


# ---------------------------------
# DOWNLOAD AND READ FITS IMAGE
# ---------------------------------

@st.cache_data(ttl=1800)
def read_fits_image(url):

    response = requests.get(
        url,
        timeout=180
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

            # Convert cube to 2D image
            while data.ndim > 2:
                data = data[data.shape[0] // 2]

            if data.ndim == 2:

                image = data.astype(np.float32)

                if np.isfinite(image).any():
                    return image

    raise ValueError(
        "No valid 2D image found in this FITS file."
    )


# ---------------------------------
# SESSION STATE
# ---------------------------------

if "archive_results" not in st.session_state:
    st.session_state.archive_results = pd.DataFrame()

if "search_errors" not in st.session_state:
    st.session_state.search_errors = []


# ---------------------------------
# SEARCH BUTTON
# ---------------------------------

if search:

    st.session_state.archive_results = pd.DataFrame()
    st.session_state.search_errors = []

    if not collections:

        st.warning("Please select at least one collection.")

    else:

        frames = []

        with st.spinner("Connecting to NASA/IPAC-IRSA archive..."):

            for collection in collections:

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

        if frames:

            st.session_state.archive_results = pd.concat(
                frames,
                ignore_index=True
            )

# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------

for error in st.session_state.search_errors:
    st.error(error)

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

    # Show available metadata columns
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
        column for column in preferred_columns
        if column in results.columns
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

    # ---------------------------------
    # IMAGE PREVIEW
    # ---------------------------------

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
            "No direct FITS access URL was returned. "
            "Please check the archive metadata."
        )

        st.write(
            "The archive search worked, but image preview "
            "requires a valid access_url."
        )

    else:

        st.success(
            f"{len(usable)} observations have direct FITS URLs."
        )

        labels = []

        for i, row in usable.iterrows():

            collection = row.get(
                "selected_collection",
                "SPHEREx"
            )

            obs_id = row.get(
                "obs_id",
                f"Observation {i+1}"
            )

            labels.append(
                f"{i+1}. {collection} — {obs_id}"
            )

        st.subheader("Compare Two Observations")

        col1, col2 = st.columns(2)

        with col1:

            index_a = st.selectbox(
                "Select Observation A",
                range(len(usable)),
                format_func=lambda i: labels[i],
                key="observation_a"
            )

        with col2:

            index_b = st.selectbox(
                "Select Observation B",
                range(len(usable)),
                index=min(1, len(usable) - 1),
                format_func=lambda i: labels[i],
                key="observation_b"
            )

        for column, index, heading in [
            (col1, index_a, "Observation A"),
            (col2, index_b, "Observation B")
        ]:

            with column:

                st.markdown(f"### {heading}")

                st.write(labels[index])

                image_url = usable.iloc[index]["preview_url"]

                try:

                    with st.spinner("Loading FITS image..."):

                        image = read_fits_image(image_url)

                    fig, ax = plt.subplots(figsize=(6, 5))

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

                    st.pyplot(fig, clear_figure=True)

                    plt.close(fig)

                except Exception as error:

                    st.error(
                        f"Image loading failed: {error}"
                    )

                st.link_button(
                    "Open Original FITS File",
                    image_url,
                    use_container_width=True
                )

# ---------------------------------
# PROJECT INFORMATION
# ---------------------------------

st.divider()

st.subheader("🚀 About MECH ORBIT")

st.write(
    """
    MECH ORBIT is a student astronomy project designed
    to explore public SPHEREx infrared observations.

    This application connects to the NASA/IPAC-IRSA archive,
    searches astronomical observations using sky coordinates,
    displays archive metadata and provides FITS image previews
    whenever direct image access is available.
    """
)

st.caption(
    "Student Prototype | Team MECH ORBIT | NASA/IPAC-IRSA"
)
