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

TAP_URL = "https://caltech.edu"
SIA_URL = "https://caltech.edu"

st.set_page_config(page_title="MECH ORBIT | SPHEREx Explorer", page_icon="🌌", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "errors" not in st.session_state:
    st.session_state.errors = list()
if "searched" not in st.session_state:
    st.session_state.searched = False


def is_html_response(raw_bytes):
    head = raw_bytes[:1000].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<title>" in head


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = requests.Session()
    session.verify = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; Astroquery/0.4.11; Python-requests)",
        "Accept": "text/csv, text/plain, */*"
    })

    # Method 1: Official IRSA TAP Query on SPHEREx tables (from IRSA documentation)
    adql_spherex = (
        f"SELECT TOP 100 "
        f"p.obs_id, p.ra AS s_ra, p.dec AS s_dec, p.time_bounds_lower, "
        f"'https://irsa.ipac.caltech.edu/' || a.uri AS access_url "
        f"FROM spherex.artifact a "
        f"JOIN spherex.plane p ON a.planeid = p.planeid "
        f"WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)"
    )

    try:
        r1 = session.post(TAP_URL, data={"QUERY": adql_spherex, "LANG": "ADQL", "FORMAT": "csv"}, timeout=(15, 60))
        if r1.status_code == 200 and r1.content and not is_html_response(r1.content):
            df1 = pd.read_csv(io.BytesIO(r1.content))
            if not df1.empty and "access_url" in [c.lower() for c in df1.columns]:
                df1.columns = [str(c).strip().lower() for c in df1.columns]
                df1["selected_collection"] = collection
                return df1
    except Exception:
        pass

    # Method 2: Official IRSA TAP Query on standard IVOA ObsCore table
    adql_obscore = (
        f"SELECT TOP 100 "
        f"obs_collection, obs_id, s_ra, s_dec, t_exptime, em_min, em_max, access_url "
        f"FROM ivoa.obscore "
        f"WHERE obs_collection LIKE '%spherex%' "
        f"AND 1 = CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )

    try:
        r2 = session.post(TAP_URL, data={"QUERY": adql_obscore, "LANG": "ADQL", "FORMAT": "csv"}, timeout=(15, 60))
        if r2.status_code == 200 and r2.content and not is_html_response(r2.content):
            df2 = pd.read_csv(io.BytesIO(r2.content))
            if not df2.empty:
                df2.columns = [str(c).strip().lower() for c in df2.columns]
                df2["selected_collection"] = collection
                return df2
    except Exception:
        pass

    # Method 3: Direct SIA2 API with strict redirect & HTML guard
    sia_params = {
        "COLLECTION": collection,
        "POS": f"circle {ra} {dec} {radius}",
        "RESPONSEFORMAT": "CSV"
    }
    r3 = session.get(SIA_URL, params=sia_params, timeout=(15, 60))
    if is_html_response(r3.content):
        raise ValueError("NASA IRSA firewall blocked cloud IP or returned HTML page.")

    df3 = pd.read_csv(io.BytesIO(r3.content), on_bad_lines="skip", engine="python")
    df3.columns = [str(c).strip().lower() for c in df3.columns]
    df3["selected_collection"] = collection
    return df3


def find_fits_urls(df):
    fits_records = list()
    for idx, row in df.iterrows():
        col_name = row.get("selected_collection", "spherex")
        obs_label = row.get("obs_id", f"Observation #{len(fits_records) + 1}")
        for cell in row.values:
            val = str(cell).strip()
            if val.startswith("http") and ".fits" in val.lower():
                fits_records.append({
                    "label": f"[{col_name}] {obs_label}",
                    "url": val
                })
                break
    return pd.DataFrame(fits_records)


@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    resp = requests.get(url, verify=False, timeout=(20, 180))
    resp.raise_for_status()
    if is_html_response(resp.content):
        raise ValueError("FITS download link returned an HTML page instead of FITS data.")
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
    options=("spherex_dr1", "spherex_qr3"),
    default=("spherex_dr1",)
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
    elif not st.session_state.errors:
        st.info("No observations found at these coordinates. Try increasing radius or changing RA/Dec.")

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
