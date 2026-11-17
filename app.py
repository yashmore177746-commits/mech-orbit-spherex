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
HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
st.set_page_config(page_title="MECH ORBIT | SPHEREx & IR Explorer", page_icon=" ", layout="wide")
# ==========================================
# DEEP SPACE BACKGROUND & GLASSMORPHISM CSS
# ==========================================
SPACE_BG_CSS = """
<style>
[data-testid="stAppViewContainer"] {
 background-image: linear-gradient(rgba(4, 7, 18, 0.74), rgba(4, 7, 18, 0.88)),
 url("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?
auto=format&fit=crop&w=1920&q=80");
 background-size: cover;
 background-position: center;
 background-attachment: fixed;
 color: #f0f4ff;
}
• 
• 
• 
• [data-testid="stSidebar"] {
 background-color: rgba(8, 13, 30, 0.88);
 backdrop-filter: blur(14px);
 border-right: 1px solid rgba(255, 255, 255, 0.14);
}
[data-testid="stHeader"] {
 background: rgba(0, 0, 0, 0);
}
div[data-testid="stMetric"] {
 background-color: rgba(15, 23, 52, 0.82);
 border: 1px solid rgba(120, 160, 255, 0.30);
 padding: 12px 16px;
 border-radius: 12px;
}
div[data-testid="stExpander"], div.stDataFrame {
 background-color: rgba(12, 18, 42, 0.78);
 border-radius: 12px;
 padding: 8px;
}
div.stButton > button {
 border-radius: 10px;
 font-weight: 600;
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
# ==========================================
# SCIENTIFIC BAND DESCRIPTIONS
# ==========================================
BAND_SCIENCE = {
 "AllWISE W1 (3.4 micron Near-IR)": {
 "feature": "Old Stars & Galactic Structure",
 "insight": "3.4 um Near-Infrared penetrates interstellar dust to reveal older, cooler star 
populations and the core structure of galaxies."
 },
 "AllWISE W2 (4.6 micron Mid-IR)": {
 "feature": "Brown Dwarfs & Warm Gas",
 "insight": "4.6 um detects cool brown dwarfs, planetary nebulae, and warm molecular gas clouds 
hidden from visible-light telescopes."
 },
 "AllWISE W3 (12 micron Mid-IR)": {
 "feature": "PAH Molecules & Stellar Nurseries",
 "insight": "12 um highlights Polycyclic Aromatic Hydrocarbons (PAHs) and glowing dust lanes where 
new stars are actively forming."
 },
 "AllWISE W4 (22 micron Far-IR)": {
 "feature": "Protostars & Hot Cosmic Dust",
 "insight": "22 um captures thermal emission from warm dust cocoons surrounding newborn protostars 
and active galactic nuclei."
 },
 "2MASS J-Band (1.25 micron Near-IR)": {
 "feature": "Dust-Piercing Stellar Photometry",
 "insight": "1.25 um cuts through galactic extinction 10x better than optical light, mapping dense 
star clusters."
 },
 "2MASS K-Band (2.17 micron Near-IR)": { "feature": "Deep Galactic Core Penetration",
 "insight": "2.17 um reveals deeply embedded young stellar objects (YSOs) inside dense molecular 
clouds."
 }
}
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
 headers = {"User-Agent": "MECH-ORBIT-SPHEREx-Explorer/2.0"}
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
 mid = data.shape[0] // 2
 data = data[mid]
 if data.ndim == 2 and np.isfinite(data).any():
 return data.astype(np.float32), (hdu.name or "PRIMARY")
 raise ValueError("No valid 2D image found in FITS file.")
def show_scientific_dashboard(url, heading, band_name):
 st.markdown(f"### {heading}")
 sci = BAND_SCIENCE.get(band_name, {"feature": "Infrared Emission", "insight": "Infrared sky 
observation."})
 st.info(f"**Astrophysical Signature ({sci['feature']}):** {sci['insight']}")
 try:
 with st.spinner("Downloading & analyzing real Infrared FITS data..."): image, extension = load_fits(url)
 finite = image[np.isfinite(image)]
 low, high = np.percentile(finite, (1, 99.5))
 if low >= high:
 low, high = float(finite.min()), float(finite.max()) + 1e-6
 mean_flux = float(np.mean(finite))
 peak_flux = float(np.max(finite))
 std_noise = float(np.std(finite))
 snr_est = round(peak_flux / std_noise, 1) if std_noise > 0 else 0.0
 m1, m2, m3, m4 = st.columns(4)
 m1.metric("Peak Pixel Flux", f"{peak_flux:.2f}")
 m2.metric("Mean Background", f"{mean_flux:.2f}")
 m3.metric("Peak SNR Estimate", f"{snr_est} sigma")
 m4.metric("FITS Extension", extension)
 col_img, col_hist = st.columns((1.3, 1.0))
 with col_img:
 fig, ax = plt.subplots(figsize=(6, 5))
 fig.patch.set_facecolor("#080c1c")
 ax.set_facecolor("#080c1c")
 norm = ImageNormalize(vmin=low, vmax=high, stretch=AsinhStretch())
 im = ax.imshow(np.ma.masked_invalid(image), origin="lower", cmap="magma", norm=norm)
 cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 cbar.ax.yaxis.set_tick_params(color="#cbd5e1")
 plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cbd5e1")
 ax.set_title("Asinh-Stretched Infrared Map", color="white", pad=10)
 ax.set_xlabel("Pixel X", color="#cbd5e1")
 ax.set_ylabel("Pixel Y", color="#cbd5e1")
 ax.tick_params(colors="#cbd5e1")
 st.pyplot(fig, clear_figure=True)
 plt.close(fig)
 with col_hist:
 fig2, ax2 = plt.subplots(figsize=(5, 5))
 fig2.patch.set_facecolor("#080c1c")
 ax2.set_facecolor("#0d132b")
 clipped = finite[(finite >= low) & (finite <= high)]
 ax2.hist(clipped, bins=45, color="#f97316", edgecolor="#fdba74", alpha=0.85)
 ax2.set_title("Pixel Intensity Distribution", color="white", pad=10)
 ax2.set_xlabel("Pixel Flux Value", color="#cbd5e1")
 ax2.set_ylabel("Pixel Count", color="#cbd5e1")
 ax2.tick_params(colors="#cbd5e1")
 ax2.grid(alpha=0.2, color="#94a3b8")
 st.pyplot(fig2, clear_figure=True)
 plt.close(fig2)
 height, width = image.shape
 st.caption(f"Resolution: {width} x {height} pixels | Stretch: Asinh | Colormap: Magma")
 st.link_button("Download Original FITS File", url, use_container_width=True)
 except Exception as err:
 st.warning(f"Image unavailable: {err}")
# ==========================================
# MAIN APP UI
# ==========================================
st.title("MECH ORBIT")
st.subheader("SPHEREx & Multi-Band Infrared Sky Explorer")
st.write("Interactive astrophysical discovery tool for analyzing real infrared FITS imagery, thermal dust 
emission, and stellar nurseries.")
st.sidebar.header("Target Coordinates")
presets = {
 "Orion Nebula (M42)": (83.8221, -5.3911), "Andromeda Galaxy (M31)": (10.6847, 41.2687),
 "Galactic Center": (266.4168, -29.0078),
 "Crab Nebula (M1)": (83.6331, 22.0145),
 "Carina Nebula (NGC 3372)": (161.2650, -59.8678),
 "Custom Coordinates": (250.0000, 5.0000)
}
choice = st.sidebar.selectbox("Quick Sky Targets:", tuple(presets.keys()))
def_ra, def_dec = presets[choice]
ra = st.sidebar.number_input("Right Ascension (RA deg)", 0.0, 360.0, def_ra, format="%.4f")
dec = st.sidebar.number_input("Declination (Dec deg)", -90.0, 90.0, def_dec, format="%.4f")
radius = st.sidebar.number_input("Search Radius (deg)", 0.05, 2.0, 0.30, step=0.05, format="%.2f")
st.sidebar.divider()
collections = st.sidebar.multiselect(
 "Select Archive Collections:",
 options=("spherex_dr1", "spherex_qr3"),
 default=("spherex_dr1",)
)
if st.sidebar.button("Search & Analyze Sky Region", type="primary", use_container_width=True):
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
 st.success(f"Loaded {len(df)} multi-band infrared FITS datasets for 
{st.session_state.target_name}.")
 st.subheader("Infrared FITS Image & Photometry Analyzer")
 idx = st.selectbox(
 f"Select Infrared Wavelength Band ({len(df)} available):",
 options=list(range(len(df))),
 format_func=lambda i: f"{df.loc[i, 'band_name']}"
 )
 band_title = df.loc[idx, "band_name"]
 show_scientific_dashboard(
 df.loc[idx, "access_url"],
 f"{st.session_state.target_name} - {band_title}",
 band_title
 )
 st.divider()
 st.subheader("Observation Metadata & FITS Links")
 st.dataframe(df, use_container_width=True)
 st.download_button("Download Archive CSV", df.to_csv(index=False).encode("utf-8"), f"mechorbit_{ra}
_{dec}.csv", "text/csv")
st.divider()
st.subheader("About MECH ORBIT")
st.write(
 "MECH ORBIT bridges public infrared sky surveys with interactive FITS visualization and automated 
photometry—empowering students and citizen scientists to explore star-forming regions, cosmic dust, and 
hidden galaxies."
)
