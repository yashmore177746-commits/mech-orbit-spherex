# MECH ORBIT — SPHEREx Sky Explorer

A student prototype that searches public SPHEREx archive observations through NASA/IPAC-IRSA and attempts to display FITS images for side-by-side comparison.

## Run locally

1. Install Python 3.10 or newer.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`

## Data source

- NASA/IPAC-IRSA SPHEREx archive: https://irsa.ipac.caltech.edu/Missions/spherex.html
- SIA service endpoint: https://irsa.ipac.caltech.edu/SIA
- Collections queried: `spherex_dr1` and `spherex_qr3`

## Important limitations

This is an initial prototype. The archive may change its available collections, metadata columns, and FITS access links. Test against live archive responses and verify image metadata before presenting. Validated moving-object detection and scientific image-difference analysis are not implemented yet.
