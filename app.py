import io
import urllib3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SIA_URL = "https://caltech.edu"

st.set_page_config(page_title="MECH ORBIT | SPHEREx Explorer", page_icon="🌌", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "errors" not in st.session_state:
    st.session_state.errors = list()
if "searched" not in st.session_state:
    st.session_state.searched = False


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"})

    params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}",
        "RESPONSEFORMAT": "CSV"
    }

    resp = session.get(SIA_URL, params=params, timeout=(20, 120))
    resp.raise_for_status()

    raw = resp.content.lstrip()
    if not raw:
        raise ValueError("Empty response from NASA archive.")
    if raw[:500].lower().startswith(b"<html") or b"proxy error" in raw[:2000].lower():
        raise ValueError("NASA archive server busy (HTML/proxy error). Try smaller radius.")

    df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip", engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["selected_collection"] = collection
    return df


def find_fits_urls(df):
    fits_records = list()
    for idx, row in df.iterrows():
        col_name = row.get("selected_collection", "spherex")
        for cell in row.values:
            val = str(cell).strip()
            if val.startswith("http") and ".fits" in val.lower():
                fits_records.append({
                    "label": f"[{col_name}] Observation #{len(fits_records) + 1}",
                    "url": val
                })
                break
    return pd.DataFrame(fits_records)


@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    resp = requests.get(url, verify=False, timeout=(20, 180))
    resp.raise_for_status()
    with fits.open(io.BytesIO(resp.content), memmap=False) as hdus:
        for hdu in hdus:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            data = np.asarray(data)
            while data.ndim > 2:
                data = data[data.shape // 2]
            if data.ndim == 2 and np.isfinite(data).any():
                return data.astype(np.float32), (hdu.name or "PRIMARY")
    raise ValueError("No valid 2D image found in FITS file.")


def show_image(url, heading):
    st.markdown(f"### {heading}")
    try:
        with st.spinner("Loading infrared image from NASA IRSA..."):
            image, extension = load_fits(url)
        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, (1, 99.5))
        if low >= high:
            low, high = float(finite.min()), float(finite.max()) + 1e-6

        fig, ax = plt.subplots(figsize=(6, 5))
        norm = ImageNormalize(vmin=low, vmax=high, stretch=AsinhStretch())
        ax.imshow(np.ma.masked_invalid(image), origin="lower", cmap="magma", norm=norm)
        ax.set_title(heading)
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.caption(f"Extension: {extension} | Size: {image.shape} x {image.shape} px")
        st.link_button("Open Original FITS", url, use_container_width=True)
    except Exception as err:
        st.warning(f"Image unavailable: {err}")


st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive (IRSA).")

st.sidebar.header("🔭 Target Coordinates")
ra = st.sidebar.number_input("Right Ascension (RA deg)", 0.0, 360.0, 127.8000, format="%.4f")
dec = st.sidebar.number_input("Declination (Dec deg)", -90.0, 90.0, 40.6000, format="%.4f")
radius = st.sidebar.number_input("Search Radius (deg)", 0.01, 2.0, 0.20, step=0.05, format="%.2f")

st.sidebar.divider()
collections = st.sidebar.multiselect(
    "Select SPHEREx Collections:",
    options=("spherex_dr1", "spherex_qr3", "spherex_dr1_deep", "spherex_dr1_cube"),
    default=("spherex_dr1", "spherex_qr3")
)

if st.sidebar.button("🚀 Search NASA Archive", type="primary", use_container_width=True):
    st.session_state.searched = True
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = list()
    frames = list()
    for col in collections:
        st.write(f"Searching: **{col}** ...")
        try:
            df_part = search_archive(col, ra, dec, radius)
            if not df_part.empty:
                frames.append(df_part)
        except Exception as e:
            st.session_state.errors.append(f"{col}: {e}")
    if frames:
        st.session_state.results = pd.concat(frames, ignore_index=True)

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    for err in st.session_state.errors:
        st.error(err)

if st.session_state.searched:
    df = st.session_state.results
    if not df.empty:
        st.success(f"NASA archive returned {len(df)} observations.")
        st.dataframe(df, use_container_width=True)

        st.download_button("📥 Download Archive CSV", df.to_csv(index=False).encode("utf-8"), f"spherex_{ra}_{dec}.csv", "text/csv")

        st.divider()
        st.subheader("🌌 SPHEREx Image Explorer")
        fits_df = find_fits_urls(df)
        if not fits_df.empty:
            idx = st.selectbox(
                f"Select FITS image ({len(fits_df)} available):",
                options=list(range(len(fits_df))),
                format_func=lambda i: fits_df.loc[i, "label"]
            )
            show_image(fits_df.loc[idx, "url"], fits_df.loc[idx, "label"])
        else:
            st.warning("No direct FITS URLs returned.")
    else:
        st.info("No observations found at these coordinates. Try increasing radius or changing RA/Dec.")

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
