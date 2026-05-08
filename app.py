"""
FloodSense - Flood Risk Early Warning Dashboard
Neural Nova | BNU Fest Hackathon
==============================================
Guideline compliance:
  ✅ No ML jargon visible to user
  ✅ Plain English + Urdu labels
  ✅ Color-coded risk badge (Green/Yellow/Orange/Red)
  ✅ Confidence score + population at risk + recommended action
  ✅ Missing/out-of-range inputs handled gracefully
  ✅ SHAP-style factor explanation
  ✅ No login or setup steps
  ✅ No heavy external dependencies at runtime
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FloodSense - Early Warning",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CONSTANTS & LOOKUP TABLES
# ─────────────────────────────────────────
DISTRICT_CONFIG = {
    "Sindh (Dadu / Jacobabad)": {
        "enc": 2,
        "avg_elevation_m": 109,
        "region": "Sindh",
        "population_at_risk": 14_563_770,
    },
    "Balochistan": {
        "enc": 0,
        "avg_elevation_m": 610,
        "region": "Balochistan",
        "population_at_risk": 9_182_616,
    },
    "Khyber Pakhtunkhwa (KP / Nowshera)": {
        "enc": 1,
        "avg_elevation_m": 285,
        "region": "KP",
        "population_at_risk": 4_350_490,
    },
}

MONSOON_MONTHS = {6, 7, 8, 9}  # June–September

SOIL_TO_MOISTURE = {
    "Dry (خشک)": 0.05,
    "Moist - damp but firm (نم)": 0.40,
    "Saturated - waterlogged (سیراب)": 0.85,
}

WATER_TO_AREA = {
    "None visible (کوئی نہیں)": 0.5,
    "Some - small puddles or streams (تھوڑا)": 5.0,
    "Significant - rivers overflowing (نمایاں)": 30.0,
    "Extensive - widespread flooding visible (وسیع)": 100.0,
}

RISK_CONFIG = {
    "Low": {
        "urdu": "کم خطرہ",
        "color": "#27AE60",
        "bg": "#EAFAF1",
        "emoji": "🟢",
        "action_en": "No immediate action required. Monitor local water levels and weather forecasts. Standard preparedness maintained.",
        "action_ur": "فوری کارروائی کی ضرورت نہیں۔ مقامی پانی کی سطح اور موسم کی پیشگوئی کی نگرانی کریں۔",
    },
    "Medium": {
        "urdu": "درمیانہ خطرہ",
        "color": "#F39C12",
        "bg": "#FEF9E7",
        "emoji": "🟡",
        "action_en": "Alert local emergency services. Identify low-lying areas and prepare evacuation routes. Check on vulnerable residents.",
        "action_ur": "مقامی ہنگامی خدمات کو الرٹ کریں۔ نشیبی علاقوں کی شناخت کریں اور انخلاء کے راستے تیار کریں۔",
    },
    "High": {
        "urdu": "زیادہ خطرہ",
        "color": "#E67E22",
        "bg": "#FEF0E7",
        "emoji": "🟠",
        "action_en": "Begin evacuating vulnerable areas and riverside settlements NOW. Open emergency shelters. Broadcast public warnings.",
        "action_ur": "کمزور علاقوں اور دریا کنارے بستیوں سے فوری انخلاء شروع کریں۔ ہنگامی پناہ گاہیں کھولیں۔",
    },
    "Critical": {
        "urdu": "انتہائی خطرہ",
        "color": "#E74C3C",
        "bg": "#FDEDEC",
        "emoji": "🔴",
        "action_en": "IMMEDIATE EVACUATION REQUIRED. Activate all emergency protocols. Contact NDMA and provincial disaster authority. Mobilise rescue teams.",
        "action_ur": "فوری انخلاء ضروری ہے۔ تمام ہنگامی پروٹوکول فعال کریں۔ این ڈی ایم اے سے رابطہ کریں۔",
    },
}

FEATURE_COLS_EXPECTED = [
    "water_area_km2",
    "water_area_change",
    "water_area_pct_change",
    "pressure",
    "temp_3day_avg",
    "wind_speed",
    "avg_elevation_m",
    "soil_3day_avg",
    "evaporation",
    "precipitation",
    "temperature",
    "humidity",
    "precip_3day_avg",
    "precip_7day_avg",
    "soil_moisture",
    "water_area_lag1",
    "water_area_lag2",
    "precip_lag1",
    "water_area_accel",
    "month",
    "is_monsoon",
    "day_of_year",
    "district_enc",
]

# Friendly display names for top factors (no ML jargon)
FACTOR_DISPLAY_NAMES = {
    "water_area_km2": "Surface water extent today",
    "water_area_pct_change": "Rate of water area increase",
    "water_area_lag1": "Surface water 1 day ago",
    "water_area_lag2": "Surface water 2 days ago",
    "water_area_accel": "Speed of water expansion",
    "water_area_change": "Change in water area",
    "precip_3day_avg": "3-day average rainfall",
    "precip_7day_avg": "7-day average rainfall",
    "precipitation": "Rainfall today",
    "precip_lag1": "Rainfall yesterday",
    "soil_moisture": "Current soil saturation",
    "soil_3day_avg": "3-day average soil saturation",
    "temperature": "Temperature today",
    "temp_3day_avg": "3-day average temperature",
    "pressure": "Atmospheric pressure",
    "humidity": "Relative humidity",
    "wind_speed": "Wind speed",
    "evaporation": "Evaporation rate",
    "avg_elevation_m": "District elevation",
    "month": "Month of year",
    "day_of_year": "Day of year",
    "is_monsoon": "Monsoon season",
    "district_enc": "District",
}


# ─────────────────────────────────────────
# LOAD ARTEFACTS (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artefacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "best_model.pkl"))
    pt = joblib.load(os.path.join(base, "power_transformer.pkl"))
    sc = joblib.load(os.path.join(base, "standard_scaler.pkl"))
    le = joblib.load(os.path.join(base, "district_encoder.pkl"))
    fcols = joblib.load(os.path.join(base, "feature_cols.pkl"))
    return model, pt, sc, le, fcols


@st.cache_resource(show_spinner=False)
def load_explainer(_model):
    return shap.TreeExplainer(_model)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def assign_risk(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    if prob < 0.50:
        return "Medium"
    if prob < 0.75:
        return "High"
    return "Critical"


def build_feature_row(
    district_key,
    date_val,
    rainfall_mm,
    soil_choice,
    water_choice,
    temperature_c,
    humidity_pct,
    pressure_hpa,
    wind_ms,
) -> pd.DataFrame:
    """Map user inputs to model feature vector."""
    d = DISTRICT_CONFIG[district_key]
    month = date_val.month
    doy = date_val.timetuple().tm_yday
    is_monsoon = int(month in MONSOON_MONTHS)
    soil_moist = SOIL_TO_MOISTURE[soil_choice]
    water_area = WATER_TO_AREA[water_choice]
    pressure_pa = pressure_hpa * 100.0  # hPa → Pa

    # Approximate rolling averages from today's reading
    precip_3d = rainfall_mm * 0.9
    precip_7d = rainfall_mm * 0.7
    temp_3d = temperature_c
    soil_3d = soil_moist

    # Lag features - approximate as same value (no historical data from user)
    water_lag1 = water_area
    water_lag2 = water_area
    precip_lag = rainfall_mm
    water_change = 0.0
    water_pct_change = 0.0
    water_accel = 0.0
    evaporation = max(0.0, temperature_c * 0.05)  # rough estimate

    row = {
        "water_area_km2": water_area,
        "water_area_change": water_change,
        "water_area_pct_change": water_pct_change,
        "pressure": pressure_pa,
        "temp_3day_avg": temp_3d,
        "wind_speed": wind_ms,
        "avg_elevation_m": d["avg_elevation_m"],
        "soil_3day_avg": soil_3d,
        "evaporation": evaporation,
        "precipitation": rainfall_mm,
        "temperature": temperature_c,
        "humidity": humidity_pct,
        "precip_3day_avg": precip_3d,
        "precip_7day_avg": precip_7d,
        "soil_moisture": soil_moist,
        "water_area_lag1": water_lag1,
        "water_area_lag2": water_lag2,
        "precip_lag1": precip_lag,
        "water_area_accel": water_accel,
        "month": month,
        "is_monsoon": is_monsoon,
        "day_of_year": doy,
        "district_enc": d["enc"],
    }
    return pd.DataFrame([row])


def validate_inputs(rainfall, temperature, humidity, pressure_hpa, wind):
    """Return list of validation messages. Empty list = all OK."""
    issues = []
    if rainfall is not None and (rainfall < 0 or rainfall > 1500):
        issues.append("Rainfall value is out of expected range (0–1500 mm).")
    if temperature is not None and (temperature < -50 or temperature > 70):
        issues.append("Temperature value is out of expected range (-50 to 70 °C).")
    if humidity is not None and (humidity < 0 or humidity > 100):
        issues.append("Humidity must be between 0 and 100 %.")
    if pressure_hpa is not None and (pressure_hpa < 800 or pressure_hpa > 1100):
        issues.append("Atmospheric pressure is out of expected range (800–1100 hPa).")
    if wind is not None and (wind < 0 or wind > 200):
        issues.append("Wind speed is out of expected range (0–200 m/s).")
    return issues


def scale_features(df_row, pt, sc, feature_cols):
    """Apply Yeo-Johnson + StandardScaler to a single-row DataFrame."""
    SKEWED = [
        "precip_lag1",
        "precipitation",
        "precip_3day_avg",
        "wind_speed",
        "precip_7day_avg",
        "water_area_pct_change",
        "soil_moisture",
        "soil_3day_avg",
        "evaporation",
        "water_area_change",
        "water_area_accel",
        "is_monsoon",
    ]
    X = df_row[feature_cols].copy().astype(float)
    # Replace any inf/-inf before transformation
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaN with column median (safe for single-row: use 0 as fallback)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(0.0)
    skewed_present = [c for c in SKEWED if c in X.columns]
    if skewed_present:
        transformed = pt.transform(X[skewed_present])
        # Guard against inf produced by Yeo-Johnson on extreme values
        transformed = np.where(np.isfinite(transformed), transformed, 0.0)
        X[skewed_present] = transformed
    return sc.transform(X), X


# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* App background */
    .stApp { background-color: #F0F4F8; }

    /* Global black text */
    body, p, span, div, li, td, th, label,
    .stMarkdown, .stCaption { color: #000 !important; }

    /* Header bar */
    .fs-header {
        background: linear-gradient(135deg, #1A5276 0%, #2471A3 100%);
        color: white !important;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .fs-header,
    .fs-header *,
    .fs-header h1,
    .fs-header h2,
    .fs-header h3,
    .fs-header p,
    .fs-header span,
    .fs-header div { color: white !important; }
    .fs-header h1 { margin: 0; font-size: 2rem; font-weight: 800; }
    .fs-header p  { margin: 4px 0 0; font-size: 1rem; opacity: 0.85; }

    /* Section card */
    .fs-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .fs-card h3 { margin-top: 0; color: #111 !important; font-size: 1.05rem; }

    /* Risk badge */
    .risk-badge {
        border-radius: 14px;
        padding: 28px 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .risk-badge .risk-level-en {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.1;
    }
    .risk-badge .risk-level-ur {
        font-size: 1.5rem;
        font-weight: 700;
        direction: rtl;
        margin-top: 4px;
    }
    .risk-badge .confidence-text {
        font-size: 1.1rem;
        margin-top: 12px;
        color: #111 !important;
    }

    /* Metric cards */
    .metric-box {
        background: #EBF5FB;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-box .metric-val { font-size: 1.6rem; font-weight: 800; color: #111 !important; }
    .metric-box .metric-lbl { font-size: 0.8rem; color: #111 !important; margin-top: 2px; }

    /* Action box */
    .action-box {
        border-left: 5px solid;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 10px;
    }
    .action-box p { margin: 0; color: #000 !important; }

    /* Disclaimer */
    .disclaimer {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.78rem;
        color: #111 !important;
        margin-top: 20px;
    }

    /* Analyse button */
    .stButton > button {
        background: linear-gradient(135deg, #1A5276, #2471A3);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 14px 36px;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Input labels */
    label { font-weight: 600 !important; }

    /* Sidebar - white text and headings */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    /* Sidebar input/dropdown text - white */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * {
        color: white !important;
    }
    /* Dropdown options popup (renders outside sidebar DOM) */
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [role="listbox"] *,
    [role="option"] {
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown(
    """
<div class="fs-header">
    <h1>🌊 FloodSense - Flood Risk Early Warning</h1>
    <p>Enter today's conditions to receive an instant flood risk assessment for your district &nbsp;|&nbsp;
    اپنے ضلع کی موجودہ صورتحال درج کریں</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────
try:
    model, pt, sc, le, feature_cols = load_artefacts()
    explainer = load_explainer(model)
    _artefacts_ok = True
except Exception as e:
    st.error(
        f"Could not load assessment system. Please ensure all model files are present. ({e})"
    )
    _artefacts_ok = False
    st.stop()

# ─────────────────────────────────────────
# INPUT FORM (sidebar)
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📝 Enter Conditions - حالات درج کریں")
with st.sidebar.form("flood_form"):
    st.markdown("**📍 Location & Date**")
    district = st.selectbox(
        "District / ضلع",
        options=list(DISTRICT_CONFIG.keys()),
        help="Select the district you are reporting from.",
    )
    report_date = st.date_input(
        "Date / تاریخ",
        value=datetime.date.today(),
        max_value=datetime.date.today(),
    )

    st.markdown("---")
    st.markdown("**🌧️ Rainfall**")
    rainfall = st.number_input(
        "Rainfall today (mm) / آج کی بارش",
        min_value=0.0,
        max_value=1500.0,
        value=None,
        step=0.5,
        placeholder="e.g. 45",
        help="Total rainfall recorded today in millimetres.",
    )
    soil_condition = st.selectbox(
        "Soil condition / مٹی کی حالت",
        options=list(SOIL_TO_MOISTURE.keys()),
        help="Describe how wet the ground currently feels.",
    )

    st.markdown("---")
    st.markdown("**💧 Water Levels**")
    visible_water = st.selectbox(
        "Visible surface water / نظر آنے والا پانی",
        options=list(WATER_TO_AREA.keys()),
        help="How much open water can you see in the area right now?",
    )

    st.markdown("---")
    st.markdown("**🌡️ Atmospheric Conditions**")
    temperature = st.number_input(
        "Temperature (°C) / درجہ حرارت",
        min_value=-50.0,
        max_value=70.0,
        value=None,
        step=0.5,
        placeholder="e.g. 35",
        help="Current air temperature in degrees Celsius.",
    )
    humidity = st.number_input(
        "Humidity (%) / نمی",
        min_value=0.0,
        max_value=100.0,
        value=None,
        step=1.0,
        placeholder="e.g. 80",
        help="Current relative humidity percentage.",
    )
    pressure_hpa = st.number_input(
        "Atmospheric pressure (hPa) / فضائی دباؤ",
        min_value=800.0,
        max_value=1100.0,
        value=None,
        step=1.0,
        placeholder="e.g. 1008",
        help="Atmospheric pressure in hPa. Typical range: 950–1025 hPa.",
    )
    wind_speed = st.number_input(
        "Wind speed (m/s) / ہوا کی رفتار",
        min_value=0.0,
        max_value=200.0,
        value=None,
        step=0.5,
        placeholder="e.g. 5",
        help="Wind speed in metres per second.",
    )

    st.markdown("")
    submitted = st.form_submit_button(
        "🔍  Assess Flood Risk  -  خطرے کا جائزہ لیں",
        width="stretch",
    )

# ─────────────────────────────────────────
# PROCESSING & RESULTS
# ─────────────────────────────────────────
if submitted:
    # 1. Check required fields
    missing_fields = []
    if rainfall is None:
        missing_fields.append("Rainfall today / آج کی بارش")
    if temperature is None:
        missing_fields.append("Temperature / درجہ حرارت")
    if humidity is None:
        missing_fields.append("Humidity / نمی")
    if pressure_hpa is None:
        missing_fields.append("Atmospheric pressure / فضائی دباؤ")
    if wind_speed is None:
        missing_fields.append("Wind speed / ہوا کی رفتار")

    if missing_fields:
        st.warning(
            "⚠️  Please fill in the following required fields before assessing:\n\n"
            + "\n".join(f"  • {f}" for f in missing_fields)
        )
        st.stop()

    # 2. Validate ranges
    issues = validate_inputs(rainfall, temperature, humidity, pressure_hpa, wind_speed)
    if issues:
        st.markdown(
            """
        <div style="background:#FFF3CD;border:1px solid #FFC107;border-radius:10px;padding:18px;margin-top:10px;">
            <b>⚠️ Insufficient data - manual assessment recommended</b><br>
            <span style="color:#555;">One or more values are outside the expected range. Please verify your readings.</span>
            <ul style="margin-top:8px;color:#444;">
        """
            + "".join(f"<li>{i}</li>" for i in issues)
            + """
            </ul>
        </div>""",
            unsafe_allow_html=True,
        )
        st.stop()

    # 3. Build feature vector
    X_raw = build_feature_row(
        district_key=district,
        date_val=report_date,
        rainfall_mm=float(rainfall),
        soil_choice=soil_condition,
        water_choice=visible_water,
        temperature_c=float(temperature),
        humidity_pct=float(humidity),
        pressure_hpa=float(pressure_hpa),
        wind_ms=float(wind_speed),
    )

    # 4. Scale & predict
    X_scaled, X_transformed = scale_features(X_raw, pt, sc, feature_cols)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    risk = assign_risk(prob)
    conf = round(prob * 100, 1)
    cfg = RISK_CONFIG[risk]
    dist_cfg = DISTRICT_CONFIG[district]

    # 5. Population at risk (scaled by risk tier)
    pop_total = dist_cfg["population_at_risk"]
    pop_scale = {"Low": 0.02, "Medium": 0.12, "High": 0.40, "Critical": 0.85}
    pop_at_risk = int(pop_total * pop_scale[risk])

    # ── RESULTS LAYOUT ──────────────────────
    st.markdown("---")
    st.markdown("## Assessment Results - نتائج")

    res_col1, res_col2 = st.columns([1.1, 1], gap="large")

    with res_col1:
        # Risk badge
        st.markdown(
            f"""
        <div class="risk-badge" style="background:{cfg['bg']};border:3px solid {cfg['color']};">
            <div style="font-size:3rem;">{cfg['emoji']}</div>
            <div class="risk-level-en" style="color:{cfg['color']};">{risk} Risk</div>
            <div class="risk-level-ur" style="color:{cfg['color']};">{cfg['urdu']}</div>
            <div class="confidence-text" style="color:#111;">
                Confidence: <strong>{conf}%</strong> &nbsp;|&nbsp; اعتماد: <strong>{conf}%</strong>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Recommended action
        st.markdown(
            f"""
        <div class="action-box" style="border-color:{cfg['color']};background:{cfg['bg']};">
            <p><strong>Recommended Action:</strong><br>{cfg['action_en']}</p>
            <p style="margin-top:10px;direction:rtl;text-align:right;"><strong>تجویز کردہ اقدام:</strong><br>{cfg['action_ur']}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with res_col2:
        # Key metrics
        st.markdown(
            f"""
        <div class="metric-box">
            <div class="metric-val">{conf}%</div>
            <div class="metric-lbl">Flood Probability &nbsp;|&nbsp; سیلاب کا امکان</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{pop_at_risk:,}</div>
            <div class="metric-lbl">Estimated People at Risk &nbsp;|&nbsp; خطرے میں آبادی (تخمینہ)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{district.split('(')[0].strip()}</div>
            <div class="metric-lbl">District &nbsp;|&nbsp; ضلع</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{'Monsoon Season 🌧️' if report_date.month in MONSOON_MONTHS else 'Dry Season ☀️'}</div>
            <div class="metric-lbl">Season &nbsp;|&nbsp; موسم</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── SHAP FACTOR EXPLANATION ─────────────
    st.markdown("---")
    st.markdown("### 🔍 Key Factors Driving This Assessment - اہم عوامل")
    st.caption(
        "The factors below show what most influenced this result. Bars pointing right increase flood risk; bars pointing left reduce it."
    )

    try:
        sv = explainer.shap_values(X_scaled)
        if isinstance(sv, list):
            sv_1 = sv[1][0]
        elif sv.ndim == 3:
            sv_1 = sv[0, :, 1]
        else:
            sv_1 = sv[0]

        shap_series = (
            pd.Series(sv_1, index=feature_cols)
            .sort_values(key=abs, ascending=False)
            .head(8)
        )

        fig, ax = plt.subplots(figsize=(8, 3.8))
        colors = [cfg["color"] if v > 0 else "#7F8C8D" for v in shap_series.values]
        display_labels = [FACTOR_DISPLAY_NAMES.get(f, f) for f in shap_series.index]
        ax.barh(
            display_labels[::-1],
            shap_series.values[::-1],
            color=colors[::-1],
            edgecolor="white",
            height=0.6,
        )
        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Influence on flood risk assessment", fontsize=9)
        ax.set_title(
            "Top factors in this assessment", fontsize=10, fontweight="bold", pad=8
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    except Exception:
        st.info("Factor breakdown not available for this input combination.")

    # ── DISCLAIMER ──────────────────────────
    st.markdown(
        f"""
    <div class="disclaimer">
        ⚠️ <strong>Advisory:</strong> This assessment is based on sensor readings and historical patterns from 2022–2024.
        It is a decision-support tool - not a substitute for professional emergency management judgement.
        Always verify with NDMA and local authorities before initiating large-scale evacuations.<br><br>
        <strong>تنبیہ:</strong> یہ جائزہ 2022–2024 کے تاریخی اعداد و شمار پر مبنی ہے اور صرف فیصلہ سازی میں معاونت کے لیے ہے۔ بڑے پیمانے پر انخلاء سے پہلے این ڈی ایم اے اور مقامی حکام سے ضرور رابطہ کریں۔
    </div>
    """,
        unsafe_allow_html=True,
    )

else:
    # ── WELCOME / IDLE STATE ─────────────────
    st.markdown(
        """
    <div class="fs-card" style="text-align:center;padding:40px;">
        <div style="font-size:3rem;">🌊</div>
        <h2 style="color:#111;margin-top:8px;">How to use FloodSense</h2>
        <p style="max-width:560px;margin:0 auto;color:#111;font-size:1rem;">
            Fill in today's weather and water conditions on the left, then press
            <strong>"Assess Flood Risk"</strong> to receive an instant colour-coded risk level,
            confidence score, and recommended action for your district.
        </p>
        <hr style="margin:24px auto;width:60%;border-color:#D6EAF8;">
        <p style="color:#111;font-size:0.9rem;max-width:520px;margin:0 auto;">
            اپنے ضلع کے موسمی اور پانی کے حالات درج کریں، پھر
            <strong>"خطرے کا جائزہ لیں"</strong> کا بٹن دبائیں۔
        </p>
        <br>
        <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;margin-top:8px;">
            <span style="background:#EAFAF1;color:#27AE60;padding:8px 18px;border-radius:20px;font-weight:700;">🟢 Low</span>
            <span style="background:#FEF9E7;color:#F39C12;padding:8px 18px;border-radius:20px;font-weight:700;">🟡 Medium</span>
            <span style="background:#FEF0E7;color:#E67E22;padding:8px 18px;border-radius:20px;font-weight:700;">🟠 High</span>
            <span style="background:#FDEDEC;color:#E74C3C;padding:8px 18px;border-radius:20px;font-weight:700;">🔴 Critical</span>
        </div>
    </div>

    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:4px;">
        <div class="fs-card" style="flex:1;min-width:200px;">
            <h3>📋 2022 Regional Impact (NDMA)</h3>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
                <tr style="background:#EBF5FB;"><th style="padding:6px;text-align:left;">Region</th><th>Deaths</th><th>Affected</th></tr>
                <tr><td style="padding:5px;">Sindh</td><td style="text-align:center;">678</td><td style="text-align:center;">14.6M</td></tr>
                <tr style="background:#F8F9FA;"><td style="padding:5px;">Balochistan</td><td style="text-align:center;">299</td><td style="text-align:center;">9.2M</td></tr>
                <tr><td style="padding:5px;">KP</td><td style="text-align:center;">306</td><td style="text-align:center;">4.4M</td></tr>
            </table>
        </div>
        <div class="fs-card" style="flex:1;min-width:200px;">
            <h3>⚡ Risk Level Guide</h3>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
                <tr style="background:#EBF5FB;"><th style="padding:6px;text-align:left;">Level</th><th>Probability</th><th>Action</th></tr>
                <tr><td style="padding:5px;color:#27AE60;font-weight:700;">Low</td><td style="text-align:center;">&lt; 25%</td><td>Monitor</td></tr>
                <tr style="background:#F8F9FA;"><td style="padding:5px;color:#F39C12;font-weight:700;">Medium</td><td style="text-align:center;">25–50%</td><td>Alert</td></tr>
                <tr><td style="padding:5px;color:#E67E22;font-weight:700;">High</td><td style="text-align:center;">50–75%</td><td>Evacuate</td></tr>
                <tr style="background:#F8F9FA;"><td style="padding:5px;color:#E74C3C;font-weight:700;">Critical</td><td style="text-align:center;">&gt; 75%</td><td>Emergency</td></tr>
            </table>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
