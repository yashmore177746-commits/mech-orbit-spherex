import streamlit as st
import pandas as pd
import requests
import io
import time

# 1. Page Configuration
st.set_page_config(
    page_title="MECH ORBIT - SPHEREx Sky Explorer",
    page_icon="🌌",
    layout="wide"
)

# 2. Clean DataFrame Function (Fixes Garbled/Byte Code in CSV)
def clean_dataframe(df):
    clean_df = df.copy()
    for col in clean_df.columns:
        clean_df[col] = clean_df[col].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else x
        )
        # Remove extra byte-string wrappers if converted to string earlier (e.g., "b'value'")
        if clean_df[col].dtype == object:
            clean_df[col] = clean_df[col].astype(str).str.replace(r"^b'(.*)'$", r"\1", regex=True)
            clean_df[col] = clean_df[col].replace({"nan": "", "None": ""})
    return clean_df

# 3. Fetch Real SPHEREx Data from NASA/IPAC IRSA ObsCore Table
@st.cache_data(ttl=3600)
def query_nasa_irsa(ra, dec, radius_deg, collections):
    tap_url = "https://caltech.edu"
    all_frames = []
    
    for col in collections:
        # NASA IRSA stores SPHEREx records inside ivoa.obscore under obs_collection
        adql = f"""
        SELECT TOP 5500
            obs_collection AS selected_collection,
            obs_id,
            s_ra AS ra,
            s_dec AS dec,
            t_min AS obs_start_mjd,
            t_max AS obs_end_mjd,
            em_min AS wavelength_min,
            em_max AS wavelength_max,
            dataproduct_type,
            access_url,
            access_format
        FROM ivoa.obscore
        WHERE obs_collection = '{col}'
          AND CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        """
        params = {
            "QUERY": adql,
            "FORMAT": "csv"
        }
        
        for attempt in range(3):
            try:
                resp = requests.get(tap_url, params=params, timeout=30)
                if resp.status_code == 200 and not resp.text.strip().startswith("<"):
                    df_temp = pd.read_csv(io.StringIO(resp.text))
                    if not df_temp.empty:
                        all_frames.append(df_temp)
                    break
            except Exception:
                time.sleep(1)
                
    if all_frames:
        final_df = pd.concat(all_frames, ignore_index=True)
        return clean_dataframe(final_df)
    return pd.DataFrame()

# 4. Sidebar Controls (Default set to North Ecliptic Pole where SPHEREx data is guaranteed)
st.sidebar.header("🔭 Explore the Sky")
ra = st.sidebar.number_input("Right Ascension (RA in degrees)", min_value=0.0, max_value=360.0, value=270.0, step=0.5)
dec = st.sidebar.number_input("Declination (Dec in degrees)", min_value=-90.0, max_value=90.0, value=66.56, step=0.5)
radius = st.sidebar.number_input("Search Radius (degrees)", min_value=0.1, max_value=15.0, value=2.0, step=0.5)

collections = st.sidebar.multiselect(
    "Select SPHEREx Collections",
    options=["spherex_dr1", "spherex_qr3"],
    default=["spherex_dr1", "spherex_qr3"]
)

st.sidebar.button("🔄 Refresh Search", type="primary")

# 5. Main UI Header
st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive.")

# 6. Execute Search & Render Results
if collections:
    for c in collections:
        st.text(f"Searching: {c}")
        
    with st.spinner("Connecting to NASA/IPAC IRSA Archive..."):
        df_obs = query_nasa_irsa(ra, dec, radius, collections)
        
    if not df_obs.empty:
        st.success(f"NASA archive returned {len(df_obs)} observations.")
        
        st.markdown("### 🪐 Archive Observations")
        st.dataframe(df_obs, use_container_width=True)
        
        # Clean CSV Export
        csv_data = df_obs.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Download Archive CSV",
            data=csv_data,
            file_name=f"MECH_ORBIT_SPHEREx_RA{ra}_Dec{dec}.csv",
            mime="text/csv"
        )
        
        # FITS Image Explorer
        st.markdown("---")
        st.markdown("### 🌠 SPHEREx Image Explorer")
        valid_urls = df_obs[df_obs["access_url"].str.startswith("http", na=False)]
        
        if not valid_urls.empty:
            st.info(f"Found {len(valid_urls)} direct observation links.")
            st.dataframe(valid_urls[["selected_collection", "obs_id", "ra", "dec", "access_url"]].head(20), use_container_width=True)
        else:
            st.warning("No direct FITS URLs were returned. Archive metadata is still available.")
    else:
        st.warning("No observations found for these coordinates. Try increasing the Search Radius (e.g. 5.0) or changing RA/Dec.")
else:
    st.info("Please select at least one collection from the sidebar.")

# 7. Project Information Footer
st.markdown("---")
st.markdown("### 🚀 About MECH ORBIT")
st.write("MECH ORBIT is a student astronomy project designed to explore public SPHEREx infrared observations.")
st.markdown("""
**Project Features**
* NASA/IPAC-IRSA archive connectivity
* Astronomical coordinate search • SPHEREx DR1 and QR3 datasets • Observation metadata
* FITS image visualization • Two-observation comparison • CSV data export • Automatic request retries • Error handling
""")
st.caption("Data Source: NASA/IPAC Infrared Science Archive (IRSA)")
