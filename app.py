import io
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch

SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
st.set_page_config(page_title="MECH ORBIT | SPHEREx Sky Explorer", page_icon="🌌", layout="wide")
st.title("🌌 MECH ORBIT — SPHEREx Sky Explorer")
st.caption("Explore public SPHEREx infrared observations from NASA/IPAC-IRSA.")

with st.sidebar:
    st.header("Search the sky")
    ra = st.number_input("Right Ascension (degrees)", 0.0, 360.0, 127.69444, 0.1)
    dec = st.number_input("Declination (degrees)", -90.0, 90.0, -39.17760, 0.1)
    radius = st.number_input("Search radius (degrees)", 0.001, 2.0, 0.05, 0.01, format="%.3f")
    collections = st.multiselect("SPHEREx collections", ["spherex_dr1", "spherex_qr3"], default=["spherex_dr1", "spherex_qr3"])
    search = st.button("🔭 Search SPHEREx images", type="primary")

@st.cache_data(ttl=3600)
def query_archive(collection, ra_value, dec_value, radius_value):
    params = {"COLLECTION": collection, "POS": f"circle {ra_value} {dec_value} {radius_value}", "RESPONSEFORMAT": "CSV"}
    response = requests.get(SIA_URL, params=params, timeout=90)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))

def get_access_url(row):
    for column in ("access_url", "cloud_access"):
        if column in row.index and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value.startswith("http"):
                return value
    return None

@st.cache_data(ttl=1800)
def read_fits_image(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with fits.open(io.BytesIO(response.content), memmap=False) as hdus:
        for hdu in hdus:
            data = getattr(hdu, "data", None)
            if data is not None and getattr(data, "ndim", 0) >= 2:
                while data.ndim > 2:
                    data = data[0]
                if data.ndim == 2:
                    return data.astype("float32")
    raise ValueError("No two-dimensional image was found in this FITS file.")

if "archive_results" not in st.session_state:
    st.session_state.archive_results = pd.DataFrame()

if search:
    if not collections:
        st.warning("Select at least one SPHEREx collection.")
    else:
        frames = []
        with st.spinner("Searching the NASA/IPAC-IRSA archive..."):
            for collection in collections:
                try:
                    frame = query_archive(collection, ra, dec, radius)
                    if not frame.empty:
                        frame["selected_collection"] = collection
                        frames.append(frame)
                except Exception as error:
                    st.error(f"{collection}: {error}")
        st.session_state.archive_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

results = st.session_state.archive_results
if results.empty:
    st.info("Enter sky coordinates in the sidebar and press Search SPHEREx images.")
else:
    st.success(f"Archive returned {len(results)} result(s).")
    st.subheader("Archive observations")
    display_columns = [c for c in ["selected_collection", "obs_id", "s_ra", "s_dec", "t_min", "t_max", "em_min", "em_max", "access_url"] if c in results.columns]
    st.dataframe(results[display_columns] if display_columns else results, use_container_width=True, hide_index=True)
    results = results.copy()
    results["preview_url"] = results.apply(get_access_url, axis=1)
    usable = results[results["preview_url"].notna()].reset_index(drop=True)
    if usable.empty:
        st.warning("No direct FITS access URL was found in the returned columns. Check archive metadata.")
    else:
        labels = [f"{i+1}. {row.get('selected_collection', 'SPHEREx')} — {row.get('obs_id', 'Observation')}" for i, row in usable.iterrows()]
        st.subheader("Compare two observations")
        left, right = st.columns(2)
        with left:
            index_a = st.selectbox("Observation A", range(len(usable)), format_func=lambda i: labels[i], key="obs_a")
        with right:
            index_b = st.selectbox("Observation B", range(len(usable)), index=min(1, len(usable)-1), format_func=lambda i: labels[i], key="obs_b")
        for column, index, heading in [(left, index_a, "Observation A"), (right, index_b, "Observation B")]:
            with column:
                st.markdown(f"**{heading}: {labels[index]}**")
                try:
                    with st.spinner("Loading FITS image..."):
                        image = read_fits_image(usable.iloc[index]["preview_url"])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    norm = ImageNormalize(image, interval=PercentileInterval(99.5), stretch=AsinhStretch())
                    ax.imshow(image, origin="lower", cmap="magma", norm=norm)
                    ax.set_xlabel("Pixel X")
                    ax.set_ylabel("Pixel Y")
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                except Exception as error:
                    st.error(f"Could not display this FITS image: {error}")
                st.link_button("Open original FITS file", str(usable.iloc[index]["preview_url"]))

st.divider()
st.markdown("### About SPHEREx\nSPHEREx is a NASA all-sky survey mission that observes the universe in near-infrared spectral bands. This app searches public archive records provided by NASA/IPAC-IRSA.")
st.caption("Student prototype by team Mech Orbit. Verify live archive responses before the final hackathon demonstration.")
