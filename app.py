import io
import csv
import urllib3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.visualization import ImageNormalize, AsinhStretch

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SIA_URL = "https://caltech.edu"
TAP_URL = "https://caltech.edu"

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide"
)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

if "errors" not in st.session_state:
    st.session_state.errors = []

def make_session():
    session = requests.Session()
    session.verify = False

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MECH-ORBIT/2.0"
    })

    return session

def parse_nasa_response(raw_bytes, collection):
    text = raw_bytes.decode("utf-8", errors="replace").strip()

    if not text:
        raise ValueError("Empty response from NASA archive.")

    lower_head = text[:2000].lower()

    if lower_head.startswith("<html") or "proxy error" in lower_head or "404 not found" in lower_head:
        raise ValueError(f"Server returned webpage/error instead of data: {text[:250]}")

    if text.startswith("<?xml") or "<votable" in lower_head:
        table = parse_single_table(io.BytesIO(raw_bytes)).to_table()
        df = table.to_pandas()
        for col in df.select_dtypes(include=("object",)).columns:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        df.columns = [str(c).strip().lower() for c in df.columns]
        df["collection"] = collection
        return df

    lines = [
        line for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "\\", "|"))
    ]

    if not lines:
        return pd.DataFrame()

    reader = csv.reader(lines)
    rows = list(reader)

    if len(rows) < 2:
        return pd.DataFrame()

    header = [h.strip().lower() for h in rows[0]]
    num_cols = len(header)

    clean_rows = []
    for r in rows[1:]:
        if len(r) == num_cols:
            clean_rows.append(r)
        elif len(r) > num_cols:
            clean_rows.append(r[:num_cols])

    df = pd.DataFrame(clean_rows, columns=header)
    df["collection"] = collection
    return df

@st.cache_data(ttl=300, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = make_session()

    adql_query = (
        f"SELECT TOP 50 obs_id, obs_collection, s_ra, s_dec, em_min, em_max, t_exptime, access_url "
        f"FROM spherex.obscore "
        f"WHERE obs_collection = '{collection}' "
        f"AND 1 = CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )

    try:
        tap_resp = session.post(
            TAP_URL,
            data={"QUERY": adql_query, "FORMAT": "csv"},
            timeout=(20, 90)
        )
        if tap_resp.status_code == 200 and tap_resp.content:
            return parse_nasa_response(tap_resp.content, collection)
    except Exception:
        pass

    sia_resp = session.get(
        SIA_URL,
        params={
            "COLLECTION": collection,
            "POS": f"circle {ra} {dec} {radius}"
        },
        timeout=(20, 90)
    )
    sia_resp.raise_for_status()
    return parse_nasa_response(sia_resp.content, collection)

@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    session = make_session()
    response = session.get(url, timeout=(20, 180))
    response.raise_for_status()

    with fits.open(io.BytesIO(response.content), memmap=False) as hdus:
        for hdu in hdus:
            data = getattr(hdu, "data", None)
            if data is None:
                continue

            data = np.asarray(data)
            if data.ndim < 2:
                continue

            while data.ndim > 2:
                mid_idx = data.shape[0] // 2
                data = data[mid_idx]

            if data.ndim == 2 and np.isfinite(data).any():
                return data.astype(np.float32), hdu.name

    raise ValueError("No valid 2D image found in FITS file.")

def show_image(url, heading):
    st.markdown(f"### {heading}")

    try:
        with st.spinner("Loading infrared image..."):
            image, extension = load_fits(url)

        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, (1, 99.5))

        if low == high:
            low = float(finite.min())
            high = float(finite.max()) + 1e-6

        fig, ax = plt.subplots(figsize=(6, 5))
        norm = ImageNormalize(vmin=low, vmax=high, stretch=AsinhStretch())

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

        height, width = image.shape
        st.caption(f"Extension: {extension} | Size: {width} × {height} pixels")
        st.link_button("Open Original FITS", url, use_container_width=True)

    except Exception as error:
        st.warning(f"Image unavailable: {error}")

with st.sidebar:
    st.header("🔭 Search the Sky")

    ra = st.number_input(
        "Right Ascension (degrees)",
        min_value=0.0,
        max_value=360.0,
        value=127.69,
        step=0.1,
        format="%.4f"
    )

    dec = st.number_input(
        "Declination (degrees)",
        min_value=-90.0,
        max_value=90.0,
        value=-39.17,
        step=0.1,
        format="%.4f"
    )

    radius = st.number_input(
        "Search Radius (degrees)",
        min_value=0.01,
        max_value=5.0,
        value=0.20,
        step=0.05,
        format="%.2f"
    )

    collections = st.multiselect(
        "SPHEREx Collections",
        options=(
            "spherex_dr1",
            "spherex_qr3",
            "spherex_dr1_deep",
            "spherex_qr3_deep"
        ),
        default=("spherex_dr1", "spherex_qr3")
    )

    search_btn = st.button("🚀 Search NASA Archive", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Clear Results", use_container_width=True)

    st.caption("Source: NASA/IPAC IRSA")

if clear_btn:
    st.cache_data.clear()
    st.session_state.results = pd.DataFrame()
    st.session_state.errors = []
    st.rerun()

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive.")

if search_btn:
    st.session_state.errors = []
    all_dfs = []

    if not collections:
        st.warning("Please select at least one SPHEREx collection.")
    else:
        for col in collections:
            st.write(f"Searching: `{col}`...")
            try:
                df = search_archive(col, ra, dec, radius)
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                st.session_state.errors.append(f"{col}: {str(e)}")

        if all_dfs:
            st.session_state.results = pd.concat(all_dfs, ignore_index=True)
        else:
            st.session_state.results = pd.DataFrame()

if st.session_state.errors:
    st.warning("Some NASA archive requests could not be completed.")
    with st.expander("View search errors"):
        for err in st.session_state.errors:
            st.error(err)

results_df = st.session_state.results

if not results_df.empty:
    st.success(f"Found {len(results_df)} observations!")

    possible_cols = ("collection", "obs_id", "s_ra", "s_dec", "em_min", "em_max", "t_exptime", "access_url")
    display_cols = [c for c in possible_cols if c in results_df.columns]

    st.dataframe(results_df[display_cols] if display_cols else results_df, use_container_width=True)

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data,
        file_name=f"spherex_ra{ra}_dec{dec}.csv",
        mime="text/csv"
    )

    if "access_url" in results_df.columns:
        valid_urls = results_df["access_url"].dropna().tolist()
        valid_urls = [u for u in valid_urls if str(u).startswith("http")]
        if valid_urls:
            st.divider()
            st.subheader("🖼️ FITS Image Visualization & Comparison")

            col1, col2 = st.columns(2)
            with col1:
                idx1 = st.selectbox(
                    "Select Observation 1",
                    range(len(valid_urls)),
                    format_func=lambda i: f"Obs #{i+1} ({results_df.iloc[i].get('collection', '')})"
                )
                show_image(valid_urls[idx1], f"Observation #{idx1+1}")

            if len(valid_urls) > 1:
                with col2:
                    idx2 = st.selectbox(
                        "Select Observation 2",
                        range(len(valid_urls)),
                        index=1,
                        format_func=lambda i: f"Obs #{i+1} ({results_df.iloc[i].get('collection', '')})"
                    )
                    show_image(valid_urls[idx2], f"Observation #{idx2+1}")
else:
    if search_btn and not st.session_state.errors:
        st.info("No observations found for these coordinates. Try increasing the Search Radius or changing RA/Dec.")
    elif not search_btn:
st.info("Enter coordinates and press Search NASA Archive.")
st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
st.caption("Data Source: NASA/IPAC Infrared Science Archive (IRSA) | Team: MECH ORBIT — Student Project")
