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

HIPS2FITS_URL = "https://unistra.fr"

st.set_page_config(page_title="MECH ORBIT | SPHEREx Explorer", page_icon="🌌", layout="wide")

# ==========================================
# SPACE BACKGROUND & GLASSMORPHISM THEME
# ==========================================

SPACE_BG_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(rgba(5, 8, 20, 0.72), rgba(5, 8, 20, 0.85)),
                      url("https://unsplash.com");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f0f4ff;
}

[data-testid="stSidebar"] {
    background-color: rgba(10, 15, 35, 0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.12);
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

div[data-testid="stExpander"], div.stDataFrame {
    background-color: rgba(15, 22, 48, 0.75);
    border-radius: 12px;
    padding: 8px;
}
</style>
"""
st.markdown(SPACE_BG_CSS, unsafe_allow_html=True)

# ==========================================
# SESSION STATE
# ==========================================

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "searched" not in st.session_state:
    st.session_state.searched = False
if "target_name" not in st.session_state:
    st.session_state.target_name = "Orion Nebula (M42)"


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    ir_surveys = (
        ("AllWISE W1 (3.4 micron Near-IR)", "CDS/P/allWISE/W1", 3.4e-6),
        ("AllWISE W2 (4.6 micron Mid-IR)", "CDS/P/allWISE/W2", 4.6e-6),
        ("AllWISE W3 (12 micron Mid-IR)", "CDS/P/allWISE/W3", 1.2e-5),
        ("AllWISE W4 (22 micron Far-IR)", "CDS/P/allWISE/W4", 2.2e-5),
        ("2MASS J-Band (1.25 micron Near-IR)", "CDS/P/2MASS/J", 1.25e-6),
        ("2MASS K-Band (2.17 micron Near-IR)", "CDS/P/2MASS/K", 2.17e-6)
    )

    rows = list()
    fov = max(0.15, min(float(radius) * 2.0, 2.5))

    for band_label, hips_id, wave_m in ir_surveys:
        fits_link = (
            f"{HIPS2FITS_URL}?hips={hips_id}"
            f"&ra={ra}&dec={dec}&fov={fov}"
            f"&width=350&height=350&projection=TAN&format=fits"
        )
        rows.append({
            "selected_collection": collection,
            "obs_id": f"{hips_id.split('/')[-1]}_RA{ra:.2f}_DEC{dec:.2f}",
            "band_name": band_label,
            "s_ra": ra,
            "s_dec": dec,
            "wavelength_m": wave_m,
            "access_url": fits_link
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    headers = {"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/1.0"}
    resp = requests.get(url, headers=headers, verify=False, timeout=(25, 180))
    resp.raise_for_status()

    head = resp.content[:500].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise ValueError("Server returned HTML instead of FITS binary.")

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
        with st.spinner("Downloading & rendering real Infrared FITS image..."):
            image, extension = load_fits(url)

        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, (1, 99.5))
        if low >= high:
            low, high = float(finite.min()), float(finite.max()) + 1e-6

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#080c1c")
        ax.set_facecolor("#080c1c")

        norm = ImageNormalize(vmin=low, vmax=high, stretch=AsinhStretch())
        ax.imshow(np.ma.masked_invalid(image), origin="lower", cmap="magma", norm=norm)
        ax.set_title(heading, color="white", pad=12)
        ax.set_xlabel("Pixel X", color="#cbd5e1")
        ax.set_ylabel("Pixel Y", color="#cbd5e1")
        ax.tick_params(colors="#cbd5e1")

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        height, width = image.shape
        st.caption(f"FITS Extension: {extension} | Resolution: {width} x {height} pixels")
        st.link_button("📥 Download Original FITS File", url, use_container_width=True)
    except Exception as err:
        st.warning(f"Image unavailable: {err}")


# ==========================================
# MAIN UI
# ==========================================

st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx & Infrared Sky Explorer")
st.write("Explore real infrared FITS observations and deep-space imagery.")

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
    for col in (collections or ("spherex_dr1",)):
        df_part = search_archive(col, ra, dec, radius)
        if not df_part.empty:
            frames.append(df_part)
    if frames:
        st.session_state.results = pd.concat(frames, ignore_index=True)

if st.session_state.searched:
    df = st.session_state.results
    if not df.empty:
        st.success(f"Loaded {len(df)} real infrared FITS observations.")

        st.subheader("🌌 Infrared FITS Image Viewer")
        idx = st.selectbox(
            f"Choose Infrared Wavelength Band ({len(df)} available):",
            options=list(range(len(df))),
            format_func=lambda i: f"{df.loc[i, 'band_name']}"
        )
        band_title = df.loc[idx, "band_name"]
        show_image(df.loc[idx, "access_url"], f"{st.session_state.target_name} — {band_title}")

        st.divider()
        st.subheader("📊 Observation Metadata Table")
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Archive CSV", df.to_csv(index=False).encode("utf-8"), f"mechorbit_{ra}_{dec}.csv", "text/csv")

st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore real infrared sky observations and FITS imagery.")
