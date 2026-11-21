# MECH ORBIT — SPHEREx Sky Explorer (Upgraded)

## What's included
- Official NASA/IPAC IRSA SIA2 search for SPHEREx DR1 Wide/Deep and QR3 Wide/Deep.
- FITS download and HDU inspection.
- Selectable image extension, FITS header display, image visualization and descriptive pixel statistics.
- Explicitly labelled synthetic demo mode; no synthetic data is ever passed off as a SPHEREx observation.
- Error handling and cached archive queries.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Data and scientific limitations
- Real mode queries IRSA's SIA2 service: `https://irsa.ipac.caltech.edu/SIA`.
- Collections used: `spherex_dr1`, `spherex_dr1_deep`, `spherex_qr3`, `spherex_qr3_deep`.
- The app inspects image planes and reports descriptive pixel statistics only.
- It does NOT perform the full SPHEREx calibration pipeline, PSF fitting, source extraction, background modelling, aperture photometry, or validated uncertainty propagation.
- A BUNIT value is shown from the FITS header when present. Do not assume a physical unit if it is absent.
- The DR1 all-sky cube is a separate collection (`spherex_dr1_cube`) and is not currently included in this image-search interface.

## Official references
- https://irsa.ipac.caltech.edu/spherex-archive-documentation/spherex-data-access/
- https://irsa.ipac.caltech.edu/data/SPHEREx/docs/overview_dr1.html
- https://irsa.ipac.caltech.edu/spherex-archive-documentation/spherex-data-releases/
