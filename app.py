import io
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from astropy.io import fits

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 MECH ORBIT — SPHEREx Sky Explorer")
st.caption("Explore real NASA SPHEREx infrared observations.")

with st.sidebar:
    st.header("🔭 Search the Sky")

    ra = st.number_input(
        "Right Ascension (degrees)",
        0.0, 360.0, 127.69444
    )

    dec = st.number_input(
        "Declination (degrees)",
        -90.0, 90.0, -39.17760
    )

    radius = st.number_input(
        "Search Radius (degrees)",
        0.001, 2.0, 0.05
    )

    collections = st.multiselect(
        "SPHEREx Collections",
        ["spherex_dr1", "spherex_qr3"],
        default=["spherex_dr1", "spherex_qr3"]
    )

    search = st.button(
        "🚀 Search SPHEREx",
        type="primary",
        use_container_width=True
    )


@st.cache_data(ttl=1800)
def query_archive(collection, ra, dec, radius):

    params = {
        "COLLECTION": collection,
        "POS": f"CIRCLE {ra} {dec} {radius}",
        "RESPONSEFORMAT": "CSV",
        "MAXREC": 200
    }

    response = requests.get(
        SIA_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    if not response.text.strip():
        return pd.DataFrame()

    return pd.read_csv(io.BytesIO(response.content))


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

            if data is None:
                continue

            data = np.squeeze(np.asarray(data))

            if data.ndim == 2:

                return data.astype("float32")

    raise ValueError("No 2D image found in FITS file.")


if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()


if search:

    frames = []

    if not collections:
        st.warning("Select at least one collection.")

    else:

        with st.spinner("Connecting to NASA/IPAC-IRSA..."):

            for collection in collections:

                try:

                    df = query_archive(
                        collection, ra, dec, radius
                    )

                    if not df.empty:
                        df["selected_collection"] = collection
                        frames.append(df)

                except Exception as error:
                    st.error(f"{collection}: {error}")

        if frames:
            st.session_state.results = pd.concat(
                frames, ignore_index=True
            )
        else:
            st.session_state.results = pd.DataFrame()


results = st.session_state.results

if results.empty:

    st.info("Enter sky coordinates and click Search SPHEREx.")

else:

    st.success(
        f"NASA Archive returned {len(results)} observations!"
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

    columns = [c for c in preferred if c in results.columns]

    st.dataframe(
        results[columns] if columns else results,
        use_container_width=True,
        hide_index=True
    )

    url_column = next(
        (c for c in ["access_url", "cloud_access"]
         if c in results.columns),
        None
    )

    if url_column is None:

        st.warning("No image URL column found in archive response.")

    else:

        usable = results[
            results[url_column].notna()
        ].copy()

        usable[url_column] = usable[url_column].astype(str)

        usable = usable[
            usable[url_column].str.startswith("http")
        ].reset_index(drop=True)

        if usable.empty:

            st.warning("No direct FITS image URLs found.")

        else:

            st.subheader("🌌 Infrared Image Viewer")

            labels = [
                f"{i+1}. {row.get('selected_collection', 'SPHEREx')} "
                f"- {row.get('obs_id', 'Observation')}"
                for i, row in usable.iterrows()
            ]

            selected = st.selectbox(
                "Select observation",
                range(len(usable)),
                format_func=lambda i: labels[i]
            )

            image_url = usable.iloc[selected][url_column]

            st.link_button(
                "Open Original FITS File",
                image_url
            )

            try:

                with st.spinner("Loading FITS image..."):

                    image_data = read_fits_image(image_url)

                finite = image_data[np.isfinite(image_data)]

                if finite.size == 0:
                    raise ValueError("No valid image pixels.")

                low, high = np.percentile(finite, [1, 99])

                fig, ax = plt.subplots(figsize=(9, 7))

                ax.imshow(
                    image_data,
                    origin="lower",
                    cmap="magma",
                    vmin=low,
                    vmax=high
                )

                ax.set_title("SPHEREx Infrared Observation")
                ax.set_xlabel("Pixel X")
                ax.set_ylabel("Pixel Y")

                st.pyplot(fig)
                plt.close(fig)

                st.success("Image loaded successfully!")

            except Exception as error:

                st.error(f"Image preview error: {error}")


st.divider()

st.header("🚀 About MECH ORBIT")

st.write("""
MECH ORBIT is a student astronomy project.

It connects to NASA/IPAC-IRSA to search public SPHEREx
observations using sky coordinates and display available
infrared FITS images.
""")

st.caption("Student Project | Team MECH ORBIT")
