"""
MECH ORBIT — SPHEREx Sky Explorer
NASA/IPAC IRSA SIA2-backed search and FITS inspection.
Real archive results are kept separate from optional demo data.
"""
from __future__ import annotations

import io
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from astropy.io import fits
from astropy.table import Table

IRSA_SIA = "https://irsa.ipac.caltech.edu/SIA"
TIMEOUT = 45

st.set_page_config(
    page_title="MECH ORBIT | SPHEREx Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@500;600;700&display=swap');
.stApp {
  background: radial-gradient(ellipse at 15% 0%, #18264b 0%, #0b1022 42%, #070b16 100%);
  color: #eef3ff;
}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#101a34,#090f20);}
h1,h2,h3 {font-family: Inter,sans-serif; color:#f2f6ff;}
.hero {
  padding: 1.4rem 1.6rem; border: 1px solid #354a78; border-radius: 20px;
  background: linear-gradient(120deg,rgba(33,57,105,.72),rgba(14,23,48,.78));
  box-shadow: 0 12px 40px rgba(0,0,0,.25); margin-bottom: 1rem;
}
.brand {font-family: Orbitron, sans-serif; letter-spacing: .14em; color:#8ddcff;}
.sub {color:#b9c9e8; font-size:1rem;}
div[data-testid="stMetric"] {
  background:rgba(20,33,65,.75); border:1px solid #2d416c;
  padding:14px 16px; border-radius:14px;
}
.stButton>button {
  border-radius:12px; border:1px solid #5377bd;
  background:linear-gradient(90deg,#2856a6,#553b9c); color:white; font-weight:700;
}
.note {color:#aebfdf;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="brand">MECH ORBIT</div>
  <h1 style="margin:.25rem 0">SPHEREx Sky Explorer</h1>
  <div class="sub">Search real NASA/IPAC SPHEREx observations • inspect FITS products • explore image statistics</div>
</div>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_data(ttl=900, show_spinner=False)
def query_sia(collection: str, ra: float, dec: float, radius_deg: float, maxrec: int = 100) -> pd.DataFrame:
    """Query the official IRSA SIA2 service and parse its VOTable response."""
    params = {
        "COLLECTION": collection,
        "POS": f"CIRCLE {ra} {dec} {radius_deg}",
        "RESPONSEFORMAT": "VOTABLE",
        "MAXREC": maxrec,
    }
    response = requests.get(IRSA_SIA, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    table = Table.read(io.BytesIO(response.content), format="votable")
    frame = table.to_pandas()
    frame.columns = [str(c).lower() for c in frame.columns]
    return frame

def safe_text(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    return str(value)

def fetch_fits(url: str) -> bytes:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Archive did not return a valid HTTPS FITS URL.")
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    if len(r.content) < 2880:
        raise ValueError("Downloaded response is too small to be a valid FITS file.")
    return r.content

def image_hdus(hdul):
    found = []
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None and np.asarray(data).ndim >= 2:
            found.append((idx, hdu.name or f"HDU {idx}", np.asarray(data)))
    return found

def extract_2d(data):
    arr = np.asarray(data, dtype=float)
    arr = np.squeeze(arr)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Selected FITS extension is not a 2D image (shape={arr.shape}).")
    return arr

def robust_stats(image):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return {}
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med)))
    sigma = 1.4826 * mad
    return {
        "pixels": int(finite.size),
        "minimum": float(np.nanmin(finite)),
        "maximum": float(np.nanmax(finite)),
        "mean": float(np.nanmean(finite)),
        "median": med,
        "std": float(np.nanstd(finite)),
        "robust_sigma_MAD": sigma,
        "p99": float(np.nanpercentile(finite, 99)),
    }

def demo_image(seed=7, size=220):
    """Clearly synthetic visualization; never represented as archive data."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    nebula = 2.2 * np.exp(-((x + .22)**2 / .16 + (y - .05)**2 / .07))
    nebula += 1.1 * np.exp(-((x - .35)**2 / .08 + (y + .28)**2 / .18))
    stars = np.zeros_like(nebula)
    for _ in range(45):
        sx, sy = rng.uniform(-.95, .95, 2)
        amp = rng.uniform(2, 9)
        stars += amp * np.exp(-((x-sx)**2 + (y-sy)**2) / rng.uniform(.00008,.0005))
    return nebula + stars + rng.normal(0, .18, (size, size))

# ---------- Sidebar ----------
st.sidebar.markdown("## 🛰️ Mission controls")
mode = st.sidebar.radio("Data source", ["NASA / IPAC real archive", "Synthetic demo (clearly labelled)"])
st.sidebar.caption("Real archive mode queries SPHEREx SIA2. Demo mode is generated and is not astronomical observation data.")

if mode == "NASA / IPAC real archive":
    release = st.sidebar.selectbox("Release", ["DR1 Wide", "DR1 Deep", "QR3 Wide", "QR3 Deep"])
    collection_map = {
        "DR1 Wide": "spherex_dr1",
        "DR1 Deep": "spherex_dr1_deep",
        "QR3 Wide": "spherex_qr3",
        "QR3 Deep": "spherex_qr3_deep",
    }
    collection = collection_map[release]
    st.sidebar.markdown("**Sky position (ICRS)**")
    preset = st.sidebar.selectbox("Quick target", ["Custom", "Galactic Center", "Orion Nebula", "Andromeda Galaxy", "Large Magellanic Cloud"])
    presets = {
        "Galactic Center": (266.4168, -29.0078),
        "Orion Nebula": (83.8221, -5.3911),
        "Andromeda Galaxy": (10.6847, 41.2690),
        "Large Magellanic Cloud": (80.8942, -69.7561),
    }
    default_ra, default_dec = presets.get(preset, (83.8221, -5.3911))
    ra = st.sidebar.number_input("Right Ascension (deg)", min_value=0.0, max_value=360.0, value=float(default_ra), step=0.01, format="%.6f")
    dec = st.sidebar.number_input("Declination (deg)", min_value=-90.0, max_value=90.0, value=float(default_dec), step=0.01, format="%.6f")
    radius = st.sidebar.slider("Search radius (deg)", min_value=0.01, max_value=1.0, value=0.10, step=0.01)
    maxrec = st.sidebar.select_slider("Maximum results", options=[10, 25, 50, 100, 200], value=50)
    run_search = st.sidebar.button("🔭 Search SPHEREx archive", use_container_width=True)
else:
    st.sidebar.markdown("**Demo settings**")
    seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=7)
    run_demo = st.sidebar.button("Generate demo image", use_container_width=True)

# ---------- Main ----------
if mode == "NASA / IPAC real archive":
    st.markdown("### Archive search")
    st.info("Searches the official IRSA SPHEREx SIA2 service. Results depend on public archive coverage for the selected sky position and release.")
    if run_search:
        try:
            with st.spinner("Querying NASA/IPAC IRSA…"):
                result = query_sia(collection, ra, dec, radius, maxrec)
            st.session_state["archive_results"] = result
            st.session_state["archive_query"] = {
                "collection": collection, "ra": ra, "dec": dec, "radius": radius
            }
            st.session_state.pop("selected_fits_bytes", None)
            st.success(f"Archive query completed: {len(result)} record(s).")
        except requests.HTTPError as e:
            st.error(f"IRSA returned an HTTP error: {e}")
        except Exception as e:
            st.error(f"Could not complete the archive query: {e}")
            st.caption("Check the internet connection, coordinates, and IRSA service status. No synthetic data has been substituted.")

    results = st.session_state.get("archive_results")
    if results is not None:
        q = st.session_state.get("archive_query", {})
        st.caption(f"Collection: {q.get('collection','—')}  |  Center: RA {q.get('ra','—')}°, Dec {q.get('dec','—')}°  |  Radius: {q.get('radius','—')}°")
        st.metric("Matching archive records", f"{len(results):,}")
        if len(results) == 0:
            st.warning("No matching SPHEREx records were returned for this position and collection. Try a larger radius or another release.")
        else:
            preferred = [c for c in ["obs_id","target_name","s_ra","s_dec","em_min","em_max","t_min","t_max","access_url"] if c in results.columns]
            display = results[preferred].copy() if preferred else results.copy()
            st.dataframe(display, use_container_width=True, hide_index=True)
            url_col = "access_url" if "access_url" in results.columns else None
            if url_col:
                valid_rows = results[results[url_col].astype(str).str.startswith("https://")]
                if len(valid_rows):
                    choices = list(valid_rows.index)
                    labels = []
                    for idx in choices:
                        row = results.loc[idx]
                        obs = safe_text(row.get("obs_id", idx))
                        target = safe_text(row.get("target_name", ""))
                        labels.append(f"{obs} — {target}" if target != "—" else obs)
                    selected_pos = st.selectbox("Select a SPHEREx FITS observation", range(len(choices)), format_func=lambda i: labels[i])
                    selected_idx = choices[selected_pos]
                    if st.button("⬇️ Download and inspect selected FITS", key="download_real_fits"):
                        try:
                            with st.spinner("Downloading FITS from IRSA…"):
                                fits_bytes = fetch_fits(str(results.loc[selected_idx, url_col]))
                            st.session_state["selected_fits_bytes"] = fits_bytes
                            st.session_state["selected_fits_label"] = labels[selected_pos]
                            st.success("FITS file downloaded.")
                        except Exception as e:
                            st.error(f"FITS download failed: {e}")
                else:
                    st.warning("The returned rows do not contain a usable HTTPS access_url.")
            else:
                st.warning("No access_url column was returned. Review the archive result fields above.")
    else:
        st.markdown("Enter sky coordinates and choose **Search SPHEREx archive** in the sidebar.")

    fits_bytes = st.session_state.get("selected_fits_bytes")
    if fits_bytes:
        st.divider()
        st.markdown("### FITS inspection")
        st.caption(f"Selected observation: {st.session_state.get('selected_fits_label','—')}")
        try:
            with fits.open(io.BytesIO(fits_bytes), memmap=False) as hdul:
                header_rows = []
                for i, hdu in enumerate(hdul):
                    header_rows.append({
                        "HDU": i, "Name": hdu.name, "Type": type(hdu).__name__,
                        "Shape": str(getattr(getattr(hdu, "data", None), "shape", None)),
                        "BUNIT": safe_text(hdu.header.get("BUNIT")),
                    })
                st.dataframe(pd.DataFrame(header_rows), use_container_width=True, hide_index=True)
                candidates = image_hdus(hdul)
                if not candidates:
                    st.warning("This FITS file has no image-like HDU with 2D+ data to display.")
                else:
                    names = [f"{i}: {name} — {data.shape}" for i, name, data in candidates]
                    selected_hdu = st.selectbox("Image extension", range(len(candidates)), format_func=lambda i: names[i])
                    hdu_idx, hdu_name, raw = candidates[selected_hdu]
                    image = extract_2d(raw)
                    header = hdul[hdu_idx].header
                    stats = robust_stats(image)
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Image dimensions", f"{image.shape[1]} × {image.shape[0]}")
                    c2.metric("Median", f"{stats['median']:.4g}")
                    c3.metric("Maximum", f"{stats['maximum']:.4g}")
                    c4.metric("BUNIT", safe_text(header.get("BUNIT")))
                    finite = image[np.isfinite(image)]
                    lo, hi = np.nanpercentile(finite, [1, 99.5])
                    fig = px.imshow(image, origin="lower", color_continuous_scale="Magma", zmin=float(lo), zmax=float(hi),
                                    labels={"x":"Pixel X","y":"Pixel Y","color":safe_text(header.get("BUNIT"))})
                    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=10,r=10,t=20,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("#### Image statistics")
                    st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)
                    st.caption("Statistics are descriptive pixel statistics of the selected FITS image plane. They are not a source catalogue, calibrated integrated photometry, or a scientific SNR measurement.")
                    with st.expander("Selected FITS header"):
                        st.code(header.tostring(sep="\\n", endcard=False), language="text")
        except Exception as e:
            st.error(f"FITS inspection error: {e}")

else:
    st.markdown("### Synthetic demo mode")
    st.warning("DEMO ONLY — this image is mathematically generated. It is not SPHEREx data, not a calibrated FITS product, and must not be presented as a NASA observation.")
    if run_demo or "demo_image" not in st.session_state:
        st.session_state["demo_image"] = demo_image(int(seed))
    img = st.session_state["demo_image"]
    stats = robust_stats(img)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Pixels", f"{stats['pixels']:,}")
    c2.metric("Median", f"{stats['median']:.3f}")
    c3.metric("Maximum", f"{stats['maximum']:.3f}")
    c4.metric("Robust scatter (MAD)", f"{stats['robust_sigma_MAD']:.3f}")
    fig = px.imshow(img, origin="lower", color_continuous_scale="Magma", labels={"x":"Pixel X","y":"Pixel Y","color":"Synthetic intensity"})
    fig.update_layout(height=600, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)

# ---------- Footer ----------
st.divider()
st.markdown("""
**Data provenance:** Real observations are queried from NASA/IPAC Infrared Science Archive (IRSA) using the SPHEREx SIA2 service.  
**Important:** DR1/QR3 spectral images are SPHEREx products in the 0.75–5.0 μm range. This app displays the selected FITS image plane and its header metadata; it does not run the full SPHEREx calibration pipeline.
""")
st.markdown("[Official SPHEREx Archive Guide](https://irsa.ipac.caltech.edu/spherex-archive-documentation/spherex-data-access/) · [SPHEREx DR1 Overview](https://irsa.ipac.caltech.edu/data/SPHEREx/docs/overview_dr1.html) · [IRSA SIA endpoint](https://irsa.ipac.caltech.edu/SIA)")
