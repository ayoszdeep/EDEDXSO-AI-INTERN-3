"""
Root proxy launcher for EDXSO School Influencer Discovery Tool.
Allows running `streamlit run app.py` directly from workspace root.
"""

import os
import sys

# Add subfolder to sys.path
subfolder = os.path.join(os.path.dirname(__file__), "edxso-school-influencer-discovery")
if subfolder not in sys.path:
    sys.path.insert(0, subfolder)

# Execute subfolder app
app_path = os.path.join(subfolder, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(code, globals())
