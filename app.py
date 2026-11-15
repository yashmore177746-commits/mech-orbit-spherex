from astropy.coordinates import SkyCoord
import astropy.units as u
import io
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    AsinhStretch
)

# ---------------- CONFIGURATION ----------------

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Sky Explorer",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 MECH ORBIT — SPHEREx Sky Explorer")

st.markdown("""
Explore public SPHEREx infrared observations
using NASA/IPAC-IRSA archive data.
""")

st.divider()

# ---------------- SIDEBAR ----------------

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
        ["spherex_dr1", "spherex_qr3"],
        default=["spherex_dr1", "spherex_qr3"]
    )

    search = st.button(
        "🚀 Search SPHEREx Images",
        type="primary",
        use_container_width=True
    )

# ---------------- NASA ARCHIVE SEARCH ----------------


@st.cache_data(ttl=3600)
def query_archive(collection, ra_value, dec_value, radius_value):

    coord = SkyCoord(
        ra_value,
        dec_value,
        unit="deg"
    )

    results = Irsa.query_sia(
        pos=(coord, radius_value * u.deg),
        collection=collection
    )

    return results.to_pandas()



# ---------------- FITS URL ----------------

def get_access_url(row):

    for column in ["access_url", "cloud_access"]:

        if column in row.index and pd.notna(row[column]):

            url = str(row[column]).strip()

            if url.startswith("http"):
                return url

    return None


# ---------------- FITS IMAGE READER ----------------

@st.cache_data(ttl=1800)
def read_fits_image(url):

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with fits.open(
        io.BytesIO(response.content),
        memmap=False
    ) as hdus:

        for hdu in hdus:

            data = getattr(hdu, "data", None)

            if data is not None and getattr(data, "ndim", 0) >= 2:

                while data.ndim > 2:
                    data = data[0]

                if data.ndim == 2:
                    return data.astype("float32")

    raise ValueError("No 2D image found in FITS file.")


# ---------------- SESSION STATE ----------------

if "archive_results" not in st.session_state:
    st.session_state.archive_results = pd.DataFrame()


# ---------------- SEARCH BUTTON ----------------

if search:

    if not collections:

        st.warning("Please select at least one collection.")

    else:

        frames = []

        with st.spinner("Connecting to NASA/IPAC-IRSA Archive..."):

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

                    st.error(f"{collection}: {error}")

        if frames:

            st.session_state.archive_results = pd.concat(
                frames,
                ignore_index=True
            )

        else:

            st.session_state.archive_results = pd.DataFrame()


# ---------------- DISPLAY RESULTS ----------------

results = st.session_state.archive_results

if results.empty:

    st.info(
        "Enter sky coordinates and click Search SPHEREx Images."
    )

else:

    st.success(
        f"NASA Archive returned {len(results)} observations!"
    )

    st.subheader("📡 Archive Observations")

    display_columns = [
        column for column in [
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
        if column in results.columns
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # ---------------- IMAGE PREVIEW ----------------

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
            "No direct FITS access URL found. "
            "Please check archive metadata."
        )

    else:

        st.subheader("🌌 Compare Infrared Observations")

        labels = [
            f"{i+1}. {row.get('selected_collection', 'SPHEREx')} "
            f"— {row.get('obs_id', 'Observation')}"
            for i, row in usable.iterrows()
        ]

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

        # ---------------- IMAGE DISPLAY ----------------

        for column, index, heading in [
            (col1, index_a, "Observation A"),
            (col2, index_b, "Observation B")
        ]:

            with column:

                st.markdown(f"**{heading}**")

                st.caption(labels[index])

                try:

                    with st.spinner("Loading FITS image..."):

                        image = read_fits_image(
                            usable.iloc[index]["preview_url"]
                        )

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
                        f"Could not display FITS image: {error}"
                    )

                st.link_button(
                    "Open Original FITS File",
                    str(usable.iloc[index]["preview_url"])
                )


# ---------------- ABOUT PROJECT ----------------

st.divider()

st.subheader("🚀 About MECH ORBIT")

st.write("""
MECH ORBIT is a student project designed to explore
public SPHEREx infrared observations.

The application connects to the NASA/IPAC-IRSA archive,
searches astronomical observations using sky coordinates,
and displays available FITS images.

SPHEREx is NASA's all-sky infrared survey mission.
""")

st.info(
    "Data source: NASA/IPAC-IRSA Public Archive"
)

st.caption(
    "Developed by Team MECH ORBIT | Student Space Research Project"
)
