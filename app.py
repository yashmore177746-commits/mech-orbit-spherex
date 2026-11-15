
import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astroquery.ipac.irsa import Irsa


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Sky Explorer",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 MECH ORBIT — SPHEREx Sky Explorer")
st.caption(
    "Explore real public SPHEREx infrared observations "
    "from NASA/IPAC-IRSA."
)

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
        value=0.05
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


# ---------------- ARCHIVE SEARCH ----------------

@st.cache_data(ttl=1800, show_spinner=False)
def query_archive(collection, ra_value, dec_value, radius_value):

    coordinate = SkyCoord(
        ra=ra_value * u.deg,
        dec=dec_value * u.deg,
        frame="icrs"
    )

    table = Irsa.query_sia(
        pos=(coordinate, radius_value * u.deg),
        collection=collection,
        maxrec=200
    )

    return table.to_pandas()


# ---------------- FITS IMAGE READER ----------------

@st.cache_data(ttl=1800, show_spinner=False)
def read_fits_image(url):

    response = requests.get(url, timeout=120)
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

            data = np.squeeze(data)

            if data.ndim == 2:

                return data.astype("float32")

    raise ValueError(
        "No 2D image found in this FITS file."
    )


# ---------------- SESSION STATE ----------------

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()


# ---------------- SEARCH BUTTON ----------------

if search:

    if not collections:

        st.warning("Please select at least one collection.")

    else:

        all_results = []

        with st.spinner(
            "Connecting to NASA/IPAC-IRSA archive..."
        ):

            for collection in collections:

                try:

                    data = query_archive(
                        collection,
                        ra,
                        dec,
                        radius
                    )

                    if not data.empty:

                        data["selected_collection"] = collection

                        all_results.append(data)

                except Exception as error:

                    st.error(
                        f"{collection}: {error}"
                    )

        if all_results:

            st.session_state.results = pd.concat(
                all_results,
                ignore_index=True
            )

        else:

            st.session_state.results = pd.DataFrame()


# ---------------- DISPLAY RESULTS ----------------

results = st.session_state.results

if results.empty:

    st.info(
        "Enter sky coordinates and click "
        "Search SPHEREx Images."
    )

else:

    st.success(
        f"NASA Archive returned {len(results)} observations!"
    )

    st.subheader("🛰️ Archive Observations")

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
        col for col in preferred_columns
        if col in results.columns
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # Find FITS URLs

    url_column = next(
        (
            col for col in ["access_url", "cloud_access"]
            if col in results.columns
        ),
        None
    )

    if url_column is None:

        st.warning(
            "No FITS URL column was returned by the archive."
        )

    else:

        usable = results.copy()

        usable[url_column] = usable[url_column].astype(str)

        usable = usable[
            usable[url_column].str.startswith("http")
        ].reset_index(drop=True)

        if usable.empty:

            st.warning(
                "No downloadable FITS URLs found."
            )

        else:

            st.subheader("🌌 Compare SPHEREx Images")

            labels = [
                f"{i + 1}. {row.get('selected_collection', 'SPHEREx')} "
                f"| {row.get('obs_id', 'Observation')}"
                for i, row in usable.iterrows()
            ]

            selected = st.selectbox(
                "Choose an observation",
                range(len(usable)),
                format_func=lambda i: labels[i]
            )

            image_url = usable.iloc[selected][url_column]

            st.write(labels[selected])

            st.link_button(
                "📥 Open Original FITS File",
                image_url
            )

            try:

                with st.spinner("Loading astronomical image..."):

                    image_data = read_fits_image(image_url)

                finite = image_data[np.isfinite(image_data)]

                if finite.size == 0:
                    raise ValueError("Image contains no valid pixels.")

                vmin, vmax = np.percentile(
                    finite,
                    [1, 99]
                )

                fig, ax = plt.subplots(figsize=(9, 7))

                ax.imshow(
                    image_data,
                    origin="lower",
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax
                )

                ax.set_title("SPHEREx Infrared Observation")
                ax.set_xlabel("Pixel X")
                ax.set_ylabel("Pixel Y")

                st.pyplot(fig)
                plt.close(fig)

                st.success("FITS image loaded successfully!")

            except Exception as error:

                st.error(
                    f"Image preview unavailable: {error}"
                )


# ---------------- ABOUT SECTION ----------------

st.divider()

st.header("🚀 About MECH ORBIT")

st.write("""
MECH ORBIT is a student astronomy project designed
to explore public SPHEREx infrared observations.

The application connects to the NASA/IPAC-IRSA archive,
searches observations using astronomical sky coordinates,
and displays available FITS image data.

SPHEREx studies the universe in near-infrared light.
""")

st.caption(
    "Student Prototype | Team MECH ORBIT"
)
