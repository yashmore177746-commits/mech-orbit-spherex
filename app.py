import io
import urllib3
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (
    ImageNormalize,
    AsinhStretch
)

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)

HIPS_URL = (
    "https://unistra.fr"
    "hips-image-services/hips2fits"
)

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx",
    page_icon="🌌",
    layout="wide"
)

BG_URL = (
    "https://unsplash.com"
    "photo-1462331940025-496dfbfc7564"
    "?auto=format&fit=crop&w=1920&q=80"
)

SPACE_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(
        rgba(4, 7, 18, 0.75),
        rgba(4, 7, 18, 0.88)
    ), url("{BG_URL}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f0f4ff;
}}
[data-testid="stSidebar"] {{
    background-color: rgba(8, 13, 30, 0.88);
    backdrop-filter: blur(14px);
}}
div[data-testid="stMetric"] {{
    background-color: rgba(15, 23, 52, 0.82);
    border: 1px solid rgba(120, 160, 255, 0.3);
    padding: 12px 16px;
    border-radius: 12px;
}}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()
if "searched" not in st.session_state:
    st.session_state.searched = False
if "target_name" not in st.session_state:
    st.session_state.target_name = "Orion Nebula (M42)"

BAND_INFO = {
    "AllWISE W1 (3.4 micron Near-IR)": (
        "Old Stars & Galactic Structure",
        "Penetrates dust to reveal older stars "
        "and core galactic structure."
    ),
    "AllWISE W2 (4.6 micron Mid-IR)": (
        "Brown Dwarfs & Warm Gas",
        "Detects cool brown dwarfs and warm "
        "molecular gas clouds."
    ),
    "AllWISE W3 (12 micron Mid-IR)": (
        "PAH Molecules & Stellar Nurseries",
        "Highlights glowing dust lanes where "
        "new stars are actively forming."
    ),
    "AllWISE W4 (22 micron Far-IR)": (
        "Protostars & Hot Cosmic Dust",
        "Captures thermal emission from dust "
        "cocoons around newborn protostars."
    ),
    "2MASS J-Band (1.25 micron Near-IR)": (
        "Dust-Piercing Stellar Photometry",
        "Cuts through galactic extinction 10x "
        "better than visible optical light."
    ),
    "2MASS K-Band (2.17 micron Near-IR)": (
        "Deep Galactic Core Penetration",
        "Reveals young stellar objects inside "
        "dense molecular clouds."
    )
}


@st.cache_data(ttl=600, show_spinner=False)
def search_archive(collection, ra, dec, radius):
    surveys = (
        ("AllWISE W1 (3.4 micron Near-IR)", "CDS/P/allWISE/W1", 3.4e-6),
        ("AllWISE W2 (4.6 micron Mid-IR)", "CDS/P/allWISE/W2", 4.6e-6),
        ("AllWISE W3 (12 micron Mid-IR)", "CDS/P/allWISE/W3", 1.2e-5),
        ("AllWISE W4 (22 micron Far-IR)", "CDS/P/allWISE/W4", 2.2e-5),
        ("2MASS J-Band (1.25 micron Near-IR)", "CDS/P/2MASS/J", 1.25e-6),
        ("2MASS K-Band (2.17 micron Near-IR)", "CDS/P/2MASS/K", 2.17e-6)
    )
    rows = list()
    fov = max(0.15, min(float(radius) * 2.0, 2.5))
    for label, hips_id, wave in surveys:
        link = (
            f"{HIPS_URL}?hips={hips_id}"
            f"&ra={ra}&dec={dec}&fov={fov}"
            "&width=350&height=350"
            "&projection=TAN&format=fits"
        )
        rows.append({
            "selected_collection": collection,
            "obs_id": f"{hips_id.split('/')[-1]}_{ra:.2f}_{dec:.2f}",
            "band_name": label,
            "s_ra": ra,
            "s_dec": dec,
            "wavelength_m": wave,
            "access_url": link
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def load_fits(url):
    hdr = {"User-Agent": "MECH-ORBIT/2.0"}
    resp = requests.get(url, headers=hdr, verify=False, timeout=(25, 180))
    resp.raise_for_status()
    head = resp.content[:500].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise ValueError("Server returned HTML instead of FITS.")
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


def show_dashboard(url, heading, band_name):
    st.markdown(f"### 🔭 {heading}")
    feat, desc = BAND_INFO.get(
        band_name,
        ("Infrared Emission", "Infrared sky observation.")
    )
    st.info(f"**🔬 {feat}:** {desc}")

    try:
        with st.spinner("Downloading & analyzing Infrared FITS..."):
            image, ext = load_fits(url)

        finite = image[np.isfinite(image)]
        low, high = np.percentile(finite, (1, 99.5))
        if low >= high:
            low = float(finite.min())
            high = float(finite.max()) + 1e-6

        mean_val = float(np.mean(finite))
        peak_val = float(np.max(finite))
        std_val = float(np.std(finite))
        snr_val = round(peak_val / std_val, 1) if std_val > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Pixel Flux", f"{peak_val:.2f}")
        c2.metric("Mean Background", f"{mean_val:.2f}")
        c3.metric("Peak SNR Estimate", f"{snr_val} σ")
        c4.metric("FITS Extension", ext)

        col1, col2 = st.columns((1.3, 1.0))

        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#080c1c")
            ax.set_facecolor("#080c1c")
            norm = ImageNormalize(vmin=low, vmax=high, stretch=AsinhStretch())
            im = ax.imshow(
                np.ma.masked_invalid(image),
                origin="lower",
                cmap="magma",
                norm=norm
            )
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cbd5e1")
            ax.set_title("Asinh Infrared Map", color="white", pad=10)
            ax.set_xlabel("Pixel X", color="#cbd5e1")
            ax.set_ylabel("Pixel Y", color="#cbd5e1")
            ax.tick_params(colors="#cbd5e1")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            fig2.patch.set_facecolor("#080c1c")
            ax2.set_facecolor("#0d132b")
            clipped = finite[(finite >= low) & (finite <= high)]
            ax2.hist(clipped, bins=45, color="#f97316", alpha=0.85)
            ax2.set_title("Pixel Intensity Histogram", color="white", pad=10)
            ax2.set_xlabel("Pixel Flux", color="#cbd5e1")
            ax2.set_ylabel("Pixel Count", color="#cbd5e1")
            ax2.tick_params(colors="#cbd5e1")
            ax2.grid(alpha=0.2, color="#94a3b8")
            st.pyplot(fig2, clear_figure=True)
            plt.close(fig2)

        h, w = image.shape
        st.caption(f"Resolution: {w} x {h} px | Stretch: Asinh | Magma")
        st.link_button("📥 Download Original FITS", url, use_container_width=True)

    except Exception as err:
        st.warning(f"Image unavailable: {err}")


st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx & Multi-Band Infrared Sky Explorer")
st.write("Interactive astrophysical discovery tool for real FITS sky imagery.")

st.sidebar.header("🔭 Target Coordinates")
presets = {
    "Orion Nebula (M42)": (83.8221, -5.3911),
    "Andromeda Galaxy (M31)": (10.6847, 41.2687),
    "Galactic Center": (266.4168, -29.0078),
    "Crab Nebula (M1)": (83.6331, 22.0145),
    "Carina Nebula (NGC 3372)": (161.2650, -59.8678),
    "Custom Coordinates": (250.0000, 5.0000)
}
choice = st.sidebar.selectbox("Quick Sky Targets:", tuple(presets.keys()))
def_ra, def_dec = presets[choice]

ra = st.sidebar.number_input("Right Ascension (RA)", 0.0, 360.0, def_ra, format="%.4f")
dec = st.sidebar.number_input("Declination (Dec)", -90.0, 90.0, def_dec, format="%.4f")
radius = st.sidebar.number_input("Search Radius (deg)", 0.05, 2.0, 0.30, step=0.05, format="%.2f")

st.sidebar.divider()
collections = st.sidebar.multiselect(
    "Select Archive Collections:",
    options=("spherex_dr1", "spherex_qr3"),
    default=("spherex_dr1",)
)

if st.sidebar.button("🚀 Search & Analyze Sky", type="primary", use_container_width=True):
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
        st.success(f"Loaded {len(df)} infrared FITS datasets for {st.session_state.target_name}.")
        st.subheader("🌌 Infrared FITS Image & Photometry Analyzer")
        idx = st.selectbox(
            f"Select Wavelength Band ({len(df)} available):",
            options=list(range(len(df))),
            format_func=lambda i: f"{df.loc[i, 'band_name']}"
        )
        band_title = df.loc[idx, "band_name"]
        show_dashboard(
            df.loc[idx, "access_url"],
            f"{st.session_state.target_name} — {band_title}",
            band_title
        )
        st.divider()
        st.subheader("📊 Observation Metadata Table")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Download Archive CSV",
df.to_csv(index=False).encode("utf-8"),
f"mechorbit_{ra}_{dec}.csv",
"text/csv"
)
st.divider()
st.subheader("🚀 About MECH ORBIT")
st.write(
"MECH ORBIT empowers students and citizen scientists to explore "
"star-forming regions and cosmic dust using real infrared FITS data."
)
