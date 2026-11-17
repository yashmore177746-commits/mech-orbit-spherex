import io
import urllib3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.visualization import ImageNormalize, AsinhStretch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SIA_URL = "https://caltech.edu"
TAP_URL = "https://caltech.edu"

st.set_page_config(page_title="MECH ORBIT | SPHEREx Explorer", page_icon="🌌", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "errors" not in st.session_state:
    st.session_state.errors = list()
if "searched" not in st.session_state:
    st.session_state.searched = False


def parse_votable(raw_bytes):
    table = parse_single_table(io.BytesIO(raw_bytes)).to_table()
    df = table.to_pandas()
    for col in df.select_dtypes(include=("object",)).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"})

    sia_params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}",
        "RESPONSEFORMAT": "application/x-votable+xml"
    }

    try:
        resp = session.get(SIA_URL, params=sia_params, timeout=(15, 60))
        raw = resp.content.lstrip()
        if resp.status_code == 200 and b"<votable" in raw[:2000].lower():
            df = parse_votable(raw)
            df["selected_collection"] = collection
            return df
    except Exception:
        pass

    adql = (
        f"SELECT TOP 200 obs_collection, obs_id, s_ra, s_dec, t_exptime, "
        f"em_min, em_max, dataproduct_type, access_format, access_url "
        f"FROM ivoa.obscore WHERE obs_collection = '{collection}' "
        f"AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) = 1"
    )
    tap_resp = session.post(TAP_URL, data={"QUERY": adql, "LANG": "ADQL", "FORMAT": "votable"}, timeout=(20, 90))
    tap_resp.raise_for_status()
    tap_raw = tap_resp.content.lstrip()

    if b"<votable" not in tap_raw[:2000].lower():
        raise ValueError("NASA archive returned an unexpected response.")

    df = parse_votable(tap_raw)
    df["selected_collection"] = collection
    return df


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
radius = st.sidebar.number_input("Search Radius (deg)", 0.01, 2.0, 0.10, step=0.05, format="%.2f")

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
        cols = [c for c in ("selected_collection", "obs_id", "s_ra", "s_dec", "t_exptime", "em_min", "em_max", "access_url") if c in df.columns]
        st.dataframe(df[cols] if cols else df, use_container_width=True)

        st.download_button("📥 Download Archive CSV", df.to_csv(index=False).encode("utf-8"), f"spherex_{ra}_{dec}.csv", "text/csv")

        st.divider()
        st.subheader("🌌 SPHEREx Image Explorer")
        if "access_url" in df.columns:
            fits_df = df[df["access_url"].astype(str).str.contains(r"https?://.*\.fits", case=False, na=False)].reset_index(drop=True)
            if not fits_df.empty:
                idx = st.selectbox(
                    f"Select FITS image ({len(fits_df)} available):",
                    options=list(range(len(fits_df))),
                    format_func=lambda i: f"[{fits_df.loc[i, 'selected_collection']}] {fits_df.loc[i].get('obs_id', f'Obs #{i+1}')}"
                )
                show_image(fits_df.loc[idx, "access_url"], f"SPHEREx Observation #{idx+1}")
            else:
                st.warning("No direct FITS URLs returned.")
    else:
        st.info("No observations found at these coordinates. Try increasing radius or changing RA/Dec.")

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
