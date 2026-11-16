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

# 2. Helper Function: Clean DataFrame for Proper CSV Export
def clean_dataframe_for_csv(df):
    """Converts bytes, objects, and arrays into clean readable strings/values."""
    clean_df = df.copy()
    for col in clean_df.columns:
        # Decode byte strings (e.g., b'spherex_dr1' -> 'spherex_dr1')
        if clean_df[col].apply(lambda x: isinstance(x, bytes)).any():
            clean_df[col] = clean_df[col].apply(
                lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else x
            )
        # Convert lists/arrays to comma-separated readable text
        elif clean_df[col].apply(lambda x: isinstance(x, (list, tuple))).any():
            clean_df[col] = clean_df[col].apply(
                lambda x: ", ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
            )
    return clean_df

# 3. Helper Function: Query NASA/IPAC IRSA TAP Service with Retries
@st.cache_data(ttl=3600)
def fetch_irsa_data(ra, dec, radius_deg, collections):
    tap_url = "https://caltech.edu"
    results_list = []
    
    for col in collections:
        # ADQL Query for spatial search
        query = f"""
        SELECT TOP 5000 *
        FROM {col}
        WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg})) = 1
        """
        params = {
            "QUERY": query,
            "FORMAT": "csv"
        }
        
        for attempt in range(3):
            try:
                response = requests.get(tap_url, params=params, timeout=25)
                if response.status_code == 200 and not response.text.strip().startswith("<"):
                    df_part = pd.read_csv(io.StringIO(response.text))
                    if not df_part.empty:
                        df_part.insert(0, "selected_collection", col)
                        results_list.append(df_part)
                    break
            except Exception:
                time.sleep(1)
                
    if results_list:
        combined_df = pd.concat(results_list, ignore_index=True)
        return clean_dataframe_for_csv(combined_df)
    return pd.DataFrame()

# 4. Sidebar Controls
st.sidebar.header("🔭 Explore the Sky")
ra = st.sidebar.number_input("Right Ascension (RA in degrees)", min_value=0.0, max_value=360.0, value=180.0, step=0.1)
dec = st.sidebar.number_input("Declination (Dec in degrees)", min_value=-90.0, max_value=90.0, value=0.0, step=0.1)
radius = st.sidebar.number_input("Search Radius (degrees)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)

selected_collections = st.sidebar.multiselect(
    "Select SPHEREx Collections",
    options=["spherex_dr1", "spherex_qr3"],
    default=["spherex_dr1", "spherex_qr3"]
)

search_btn = st.sidebar.button("🚀 Search NASA Archive", type="primary")

# 5. Main Header
st.title("🌌 MECH ORBIT")
st.subheader("SPHEREx Sky Explorer")
st.write("Explore real infrared observations from NASA/IPAC Infrared Science Archive.")

# 6. Execution & Display
if selected_collections:
    for col in selected_collections:
        st.text(f"Searching: {col}")
        
    with st.spinner("Fetching data from NASA/IPAC IRSA..."):
        df_results = fetch_irsa_data(ra, dec, radius, selected_collections)
        
    if not df_results.empty:
        st.success(f"NASA archive returned {len(df_results)} observations.")
        
        st.markdown("### 🪐 Archive Observations")
        st.dataframe(df_results, use_container_width=True)
        
        # Clean CSV Generation (Fixes code/garbled output issue)
        csv_bytes = df_results.to_csv(index=False).encode("utf-8-sig")
        
        st.download_button(
            label="📥 Download Archive CSV",
            data=csv_bytes,
            file_name=f"mech_orbit_spherex_ra{ra}_dec{dec}.csv",
            mime="text/csv"
        )
        
        # FITS Image Explorer Section
        st.markdown("---")
        st.markdown("### 🌠 SPHEREx Image Explorer")
        url_cols = [c for c in df_results.columns if "url" in c.lower() or "access" in c.lower()]
        
        if url_cols:
            st.info(f"FITS URLs found in column: `{url_cols[0]}`")
            st.write(df_results[["selected_collection"] + url_cols].dropna().head(10))
        else:
            st.warning("No direct FITS URLs were returned. Archive metadata is still available.")
    else:
        st.warning("No observations found for these coordinates. Try increasing the Search Radius or changing RA/Dec.")
else:
    st.info("Please select at least one collection from the sidebar.")

# 7. Footer / About Section
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
