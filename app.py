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

st.set_page_config(page_title="MECH ORBIT | SPHEREx Explorer", page_icon="🌌", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "searched" not in st.session_state:
    st.session_state.searched = False
if "source_note" not in st.session_state:
    st.session_state.source_note = ""
if "target_name" not in st.session_state:
    st.session_state.target_name = "Orion Nebula (M42)"


def is_html_response(raw_bytes):
    head = raw_bytes[:1000].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<title>" in head


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    session = requests.Session()
    session.verify = False
    session.headers.update({"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"})

    adql = (
        f"SELECT TOP 50 obs_collection, obs_id, s_ra, s_dec, t_exptime, em_min, em_max, access_url "
        f"FROM ivoa.obscore WHERE obs_collection LIKE '%spherex%' "
        f"AND 1 = CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius}))"
    )
    try:
        r1 = session.post(TAP_URL, data={"QUERY": adql, "LANG": "ADQL", "FORMAT": "csv"}, timeout=(10, 25))
        if r1.status_code == 200 and r1.content and not is_html_response(r1.content):
            df1 = pd.read_csv(io.BytesIO(r1.content))
            if not df1.empty:
                df1.columns = [str(c).strip().lower() for c in df1.columns]
                df1["selected_collection"] = collection
                return df1, "Direct NASA/IPAC IRSA SPHEREx Archive"
    except Exception:
        pass

    ir_bands = (
        ("WISE 3.4 micron (Near-IR)", "WISE 3.4", 3.4e-6),
        ("WISE 4.6 micron (Mid-IR)", "WISE 4.6", 4.6e-6),
        ("WISE 12 micron (Mid-IR)", "WISE 12", 1.2e-5),
        ("2MASS-J (1.25 micron Near-IR)", "2MASS-J", 1.25e-6),
        ("2MASS-K (2.17 micron Near-IR)", "2MASS-K", 2.17e-6)
    )

    rows = list()
    fov = max(0.20, min(float(radius) * 2.0, 2.0))
    for band_label, survey_code, wave_m in ir_bands:
        fits_link = (
            f"https://nasa.gov?"
            f"Position={ra},{dec}&Survey={survey_code}&Size={fov}&Pixels=300&Return=FITS"
        )
        rows.append({
            "selected_collection": f"{collection} (NASA IR SkyView)",
            "obs_id": f"{survey_code}_RA{ra:.2f}_DEC{dec:.2f}",
            "band_name": band_label,
            "s_ra": ra,
            "s_dec": dec,
            "em_min": wave_m,
            "access_url": fits_link
        })

    return pd.DataFrame(rows), "NASA Goddard SkyView Infrared FITS Service"


@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    resp = requests.get(url, verify=False, timeout=(25, 180))
    resp.raise_for_status()
    if is_html_response(resp.content):
        raise ValueError("Archive returned HTML instead of FITS binary.")
    with fits.open(io.BytesIO(resp.content), memmap=False) as hdus:
        for hdu in hdus:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            data = np.asarray(data)
            while data.ndim > 2:
                mid = data.shape // 2
                data = data[mid]
            if data.ndim == 2 and np.isfinite(data).any():
                return data.astype(np.float32), (hdu.name or "PRIMARY")
    raise ValueError("No valid 2D image found in FITS file.")


def show_image(url, heading):
    st.markdown(f"### 🔭 {heading}")
    try:
        with st.spinner("Downloading & rendering NASA FITS infrared photo..."):
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

        height, width = image.shape
        st.caption(f"FITS Extension: {extension} | Resolution: {width} x {height} pixels")
        st.link_button("📥 Download Original NASA FITS File", url, use_container_width=True)
    except Exception as err:
        st.warning(f"Image unavailable: {err}")


st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx & Infrared Sky Explorer")
st.write("Explore real infrared FITS observations and space imagery from NASA archives.")

st.sidebar.header("🔭 Target Coordinates")

presets = {
    "Orion Nebula (M42)": (83.8221, -5.3911),
    "Andromeda Galaxy (M31)": (10.6847, 41.2687),
    "Galactic Center": (266.4168, -29.0078),
    "Crab Nebula (M1)": (83.6331, 22.0145),
    "Custom Coordinates": (250.0000, 5.0000)
}
choice = st.sidebar.selectbox("Quick Sky Targets:", tuple(presets.keys()))
def_ra, def_dec = presets[choice]

ra = st.sidebar.number_input("Right Ascension (RA deg)", 0.0, 360.0, def_ra, format="%.4f")
dec = st.sidebar.number_input("Declination (Dec deg)", -90.0, 90.0, def_dec, format="%.4f")
radius = st.sidebar.number_input("Search Radius (deg)", 0.05, 2.0, 0.30, step=0.05, format="%.2f")

st.sidebar.divider()
collections = st.sidebar.multiselect(
    "Select Infrared Collections:",
    options=("spherex_dr1", "spherex_qr3"),
    default=("spherex_dr1",)
)

if st.sidebar.button("🚀 Search NASA Archive", type="primary", use_container_width=True):
    st.session_state.searched = True
    st.session_state.results = pd.DataFrame()
    st.session_state.target_name = choice
    frames = list()
    note = ""
    for col in (collections or ("spherex_dr1",)):
        df_part, note = search_archive(col, ra, dec, radius)
        if not df_part.empty:
            frames.append(df_part)
    if frames:
        st.session_state.results = pd.concat(frames, ignore_index=True)
        st.session_state.source_note = note

if st.session_state.searched:
    df = st.session_state.results
    if not df.empty:
        st.success(f"Loaded {len(df)} real infrared FITS observations ({st.session_state.source_note}).")

        # फोटो सर्वात वर लगेच दाखवणे (Image Explorer First)
        st.subheader("🌌 Infrared FITS Image Viewer")
        idx = st.selectbox(
            f"Choose Infrared Wavelength Band ({len(df)} available):",
            options=list(range(len(df))),
            format_func=lambda i: f"{df.loc[i].get('band_name', df.loc[i].get('obs_id', f'Obs #{i+1}'))}"
        )
        band_title = df.loc[idx].get("band_name", df.loc[idx].get("obs_id", f"Obs #{idx+1}"))
        show_image(df.loc[idx, "access_url"], f"{st.session_state.target_name} — {band_title}")

        st.divider()
        st.subheader("📊 Observation Metadata Table")
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Archive CSV", df.to_csv(index=False).encode("utf-8"), f"mechorbit_{ra}_{dec}.csv", "text/csv")

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore real NASA infrared sky observations and FITS imagery.")
