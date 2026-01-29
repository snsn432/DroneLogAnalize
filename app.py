import streamlit as st
import pandas as pd
from pymavlink import mavutil
import tempfile
import os
import io
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import pydeck as pdk
from openai import OpenAI
from fpdf import FPDF

# 🔑 [USER SETTING] Paste your OpenAI API Key here (starts with sk-...)
MY_API_KEY = "sk-REPLACE_ME_WITH_YOUR_OPENAI_KEY"

# Font settings
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Constants
GPS_MESSAGE_TYPES = ("GPS", "GPS2", "GPS_RAW", "GPS_RAW_INT", "GPS2_RAW")
MAX_CONSECUTIVE_NONE = 500

# 1. Page layout & header
st.set_page_config(page_title="ArduLogAnalyzer", layout="wide")

# 1. Auto-Connect if Key is provided in secrets/code
OPENAI_SECRET_KEY = st.secrets.get("OPENAI_API_KEY", "")
effective_api_key = OPENAI_SECRET_KEY or MY_API_KEY

if effective_api_key and effective_api_key.startswith("sk-"):
    try:
        if not effective_api_key.isascii():
            st.error("OpenAI API Key에 ASCII 이외의 문자가 포함되어 있습니다.")
        elif "openai_client" not in st.session_state:
            st.session_state.openai_client = OpenAI(api_key=effective_api_key)
    except Exception as e:
        st.error(f"API Key Error: {e}")

# --- CSS 복구: 사이드바 버튼 살리기 ---
hide_st_style = """
<style>
    /* 1. 하단 푸터(Made with Streamlit) 숨기기 */
    footer {visibility: hidden;}
    
    /* 2. 우측 하단 뱃지 숨기기 */
    .viewerBadge_container__1QS1n {display: none !important;}
    div[class*="viewerBadge"] {display: none !important;}

    /* 2-1. 우측 하단 프로필/Streamlit 버튼 숨기기 */
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDeployButton"] {display: none !important;}
    
    /* 3. 상단 헤더는 건드리지 않음 (주석 처리) -> 화살표 나옴 */
    /* header {visibility: hidden;} */

    /* Chat input fixed to bottom */
    [data-testid="stChatInput"] {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        background: white;
        padding: 0.5rem 1rem 0.75rem;
        z-index: 9999;
        box-shadow: 0 -6px 16px rgba(0, 0, 0, 0.12);
    }
    /* Make input narrower and centered */
    [data-testid="stChatInput"] > div {
        max-width: 860px;
        margin: 0 auto;
    }
    /* Avoid sidebar overlap when expanded */
    [data-testid="stSidebar"][aria-expanded="true"] ~ div [data-testid="stChatInput"] {
        left: 21rem;
    }
    [data-testid="stSidebar"][aria-expanded="false"] ~ div [data-testid="stChatInput"] {
        left: 0;
    }
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("ArduLogAnalyzer")
st.write("Upload an ArduPilot log file (.bin) to inspect basic info and vibration data.")

# Pro license state
if "is_pro" not in st.session_state:
    st.session_state.is_pro = False

# --- Helper Functions for Auth ---
DB_FILE = "users.json"
AUTO_LOGIN_FILE = "auto_login.json"

def load_users():
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE, "r") as f:
        return json.load(f)

def _normalize_user_record(record):
    if isinstance(record, dict):
        return record
    return {"password_hash": record, "pro": False}

def _save_users(users):
    with open(DB_FILE, "w") as f:
        json.dump(users, f)

def load_auto_login_user():
    if not os.path.exists(AUTO_LOGIN_FILE):
        return None
    try:
        with open(AUTO_LOGIN_FILE, "r") as f:
            data = json.load(f)
        return data.get("username")
    except Exception:
        return None

def save_auto_login_user(username):
    try:
        with open(AUTO_LOGIN_FILE, "w") as f:
            json.dump({"username": username}, f)
    except Exception:
        pass

def clear_auto_login_user():
    try:
        if os.path.exists(AUTO_LOGIN_FILE):
            os.remove(AUTO_LOGIN_FILE)
    except Exception:
        pass

def save_user(username, password):
    users = load_users()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    users[username] = {"password_hash": hashed_pw, "pro": False}
    _save_users(users)

def verify_login(username, password):
    users = load_users()
    if username not in users:
        return False
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    record = _normalize_user_record(users[username])
    return record.get("password_hash") == hashed_pw

def is_user_pro(username):
    users = load_users()
    if username not in users:
        return False
    record = _normalize_user_record(users[username])
    return bool(record.get("pro"))

def set_user_pro(username, value):
    users = load_users()
    if username not in users:
        return
    record = _normalize_user_record(users[username])
    record["pro"] = bool(value)
    users[username] = record
    _save_users(users)

with st.sidebar:
    # --- Sidebar Authentication UI ---
    st.title("Drone A.I.")

    # Initialize Session State
    if "is_logged_in" not in st.session_state:
        st.session_state["is_logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""

    # Try auto login if available
    if not st.session_state["is_logged_in"]:
        auto_user = load_auto_login_user()
        if auto_user and auto_user in load_users():
            st.session_state["is_logged_in"] = True
            st.session_state["username"] = auto_user
            st.session_state["is_pro"] = is_user_pro(auto_user)

    # If NOT logged in -> Show Login/Signup Menu
    if not st.session_state["is_logged_in"]:
        menu = st.radio("Menu", ["Login", "Sign Up"], horizontal=True)

        if menu == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            auto_login_opt_in = st.checkbox("Remember me", value=True)
            if st.button("Login"):
                if verify_login(username, password):
                    st.session_state["is_logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["is_pro"] = is_user_pro(username)
                    if auto_login_opt_in:
                        save_auto_login_user(username)
                    else:
                        clear_auto_login_user()
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")

        elif menu == "Sign Up":
            st.subheader("Create New Account")
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")

            if st.button("Sign Up"):
                users = load_users()
                if new_pass != confirm_pass:
                    st.error("❌ Passwords do not match.")
                elif new_user in users:
                    st.error("❌ Username already exists.")
                elif len(new_pass) < 4:
                    st.warning("⚠️ Password must be at least 4 characters.")
                else:
                    save_user(new_user, new_pass)
                    st.success("✅ Account created! Please go to Login.")

    # If Logged In -> Show User Info & Logout
    else:
        st.success(f"👤 User: **{st.session_state['username']}**")
        if st.button("Log out"):
            st.session_state["is_logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["is_pro"] = False
            clear_auto_login_user()
            st.rerun()
        st.divider()

    st.header("🔑 Pro License Key")
    if st.session_state.get("is_logged_in", False):
        user_is_pro = is_user_pro(st.session_state["username"])
        st.session_state.is_pro = user_is_pro

        if user_is_pro:
            st.success("✅ Pro features unlocked!")
        else:
            # --- License Key Section with Toggle ---
            col1, col2 = st.columns([3, 1.5])  # Arrange input and toggle nicely

            with col1:
                # 1. The Input Field (Standard text input)
                license_key = st.text_input("Pro License Key", key="license_input", placeholder="Enter key here")

            with col2:
                st.write("")  # Spacer
                st.write("")  # Spacer
                # 2. Toggle Button
                show_key = st.checkbox("👁️ Show", value=False)

            # 3. Conditional CSS Masking
            # Only apply the "dots" style if the user wants to HIDE the key
            if not show_key:
                st.markdown("""
                <style>
                    /* Hide text in the specific input field */
                    input[aria-label="Pro License Key"] {
                        -webkit-text-security: disc !important;
                        text-security: disc !important;
                        font-family: text-security-disc !important;
                    }
                </style>
                """, unsafe_allow_html=True)

            pro_password = st.secrets.get("PRO_PASSWORD", None)
            if license_key and pro_password and license_key == pro_password:
                set_user_pro(st.session_state["username"], True)
                st.session_state.is_pro = True
                st.success("✅ Pro features unlocked!")
            else:
                st.info("🔒 Advanced features are locked.")
                st.link_button(
                    "🚀 Get Lifetime Access ($9)",
                    "https://scownu.gumroad.com/l/fdyrdb",
                )
    else:
        st.info("🔒 **Pro Features are locked.** Please log in to enter a license key.")
        st.session_state.is_pro = False


def _safe_getattr(msg, *attrs):
    """Safely get attribute from message with multiple fallback names."""
    for attr in attrs:
        val = getattr(msg, attr, None)
        if val is not None:
            return val
    return None

def _extract_time_us(msg):
    """Extract timestamp in microseconds from a log message."""
    time_us = _safe_getattr(msg, "TimeUS", "time_us", "time_usec", "timeUS")
    if time_us is not None:
        try:
            return float(time_us)
        except (TypeError, ValueError):
            return None
    time_ms = _safe_getattr(msg, "time_boot_ms", "TimeMS", "time_ms")
    if time_ms is not None:
        try:
            return float(time_ms) * 1000.0
        except (TypeError, ValueError):
            return None
    return None

def generate_log_summary(file_path: str) -> str:
    """Parse the log with pymavlink and return a concise summary string for AI context injection."""
    att_roll_errors = []
    att_pitch_errors = []
    nsats_min = None
    hdop_max = None
    err_messages = []
    ev_messages = []
    ctnu_tho_values = []
    vibe_x_vals = []
    vibe_y_vals = []
    vibe_z_vals = []
    bat_voltages = []

    master = mavutil.mavlink_connection(file_path)
    consecutive_none = 0

    while True:
        msg = master.recv_match(blocking=False)
        if msg is None:
            consecutive_none += 1
            if consecutive_none >= MAX_CONSECUTIVE_NONE:
                break
            continue

        consecutive_none = 0
        mtype = msg.get_type()

        if mtype == "ATT":
            des_roll = _safe_getattr(msg, "DesRoll", "desroll")
            roll = _safe_getattr(msg, "Roll", "roll")
            des_pitch = _safe_getattr(msg, "DesPitch", "despitch")
            pitch = _safe_getattr(msg, "Pitch", "pitch")
            if des_roll is not None and roll is not None:
                att_roll_errors.append(abs(float(des_roll) - float(roll)))
            if des_pitch is not None and pitch is not None:
                att_pitch_errors.append(abs(float(des_pitch) - float(pitch)))

        elif mtype in GPS_MESSAGE_TYPES:
            nsats = _safe_getattr(msg, "NSats", "nsats")
            hdop = _safe_getattr(msg, "HDop", "HDOP", "hdop")
            if nsats is not None:
                nsats_min = nsats if nsats_min is None else min(nsats_min, int(nsats))
            if hdop is not None:
                hdop_max = hdop if hdop_max is None else max(hdop_max, float(hdop))

        elif mtype == "ERR":
            err_messages.append(str(msg))
        elif mtype == "EV":
            ev_messages.append(str(msg))
        elif mtype == "CTUN":
            tho = _safe_getattr(msg, "ThO", "tho", "THO")
            if tho is not None:
                ctnu_tho_values.append(float(tho))
        elif mtype == "VIBE":
            vx = _safe_getattr(msg, "VibeX", "vibe_x")
            vy = _safe_getattr(msg, "VibeY", "vibe_y")
            vz = _safe_getattr(msg, "VibeZ", "vibe_z")
            if vx is not None:
                vibe_x_vals.append(float(vx))
            if vy is not None:
                vibe_y_vals.append(float(vy))
            if vz is not None:
                vibe_z_vals.append(float(vz))
        elif mtype == "BAT":
            volt = _safe_getattr(msg, "Volt", "volt", "V")
            if volt is not None:
                bat_voltages.append(float(volt))

    parts = []
    if att_roll_errors:
        parts.append(f"ATT: DesRoll-Roll mean error={sum(att_roll_errors) / len(att_roll_errors):.3f}")
    if att_pitch_errors:
        parts.append(f"ATT: DesPitch-Pitch mean error={sum(att_pitch_errors) / len(att_pitch_errors):.3f}")
    if nsats_min is not None:
        parts.append(f"GPS: NSats min={nsats_min}")
    if hdop_max is not None:
        parts.append(f"GPS: HDOP max={hdop_max:.2f}")
    if err_messages:
        parts.append(f"ERR messages (10 of {len(err_messages)}): " + " | ".join(err_messages[:10]))
    if ev_messages:
        parts.append(f"EV messages (10 of {len(ev_messages)}): " + " | ".join(ev_messages[:10]))
    if ctnu_tho_values:
        parts.append(f"CTUN: ThO mean={sum(ctnu_tho_values) / len(ctnu_tho_values):.2f}, max={max(ctnu_tho_values):.2f}")
    if vibe_x_vals:
        parts.append(f"VIBE: VibeX mean={sum(vibe_x_vals)/len(vibe_x_vals):.2f}, max={max(vibe_x_vals):.2f}")
    if vibe_y_vals:
        parts.append(f"VIBE: VibeY mean={sum(vibe_y_vals)/len(vibe_y_vals):.2f}, max={max(vibe_y_vals):.2f}")
    if vibe_z_vals:
        parts.append(f"VIBE: VibeZ mean={sum(vibe_z_vals)/len(vibe_z_vals):.2f}, max={max(vibe_z_vals):.2f}")
    if bat_voltages:
        parts.append(f"BAT: start voltage={bat_voltages[0]:.2f}V, min voltage={min(bat_voltages):.2f}V")

    return " | ".join(parts) if parts else "No summarizable data found in the log."


def _format_duration(seconds_value):
    if seconds_value is None:
        return "N/A"
    try:
        total_seconds = int(round(float(seconds_value)))
    except (TypeError, ValueError):
        return "N/A"
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def _safe_pdf_text(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _safe_unicode_text(text):
    """Ensure text is valid UTF-8 for API payloads."""
    if text is None:
        return ""
    return str(text).encode("utf-8", "ignore").decode("utf-8")


def build_report_summary(data, uploaded_file):
    df_bat = data.get("df_bat")
    df_vibe = data.get("df_vibe")
    duration_seconds = data.get("duration_seconds")
    if duration_seconds is None:
        if df_bat is not None and len(df_bat) > 0:
            duration_seconds = float(df_bat["TimeS"].max()) if "TimeS" in df_bat.columns else float(df_bat["time_normalized"].max())
        elif df_vibe is not None and len(df_vibe) > 0:
            duration_seconds = float(df_vibe["TimeS"].max()) if "TimeS" in df_vibe.columns else float(df_vibe["time_normalized"].max())

    max_current = None
    max_voltage = None
    if df_bat is not None and len(df_bat) > 0:
        max_current = float(df_bat["curr"].max())
        max_voltage = float(df_bat["volt"].max())

    return {
        "flight_duration": _format_duration(duration_seconds),
        "max_altitude": "N/A",
        "max_current": f"{max_current:.2f} A" if max_current is not None else "N/A",
        "max_voltage": f"{max_voltage:.2f} V" if max_voltage is not None else "N/A",
        "log_file_name": getattr(uploaded_file, "name", "Unknown"),
    }


def get_error_desc(err_type, err_msg):
    full_text = f"{err_type} {err_msg}".upper()
    translations = {
        "CRASH": "💥 Crash Detected (Check motors/frame)",
        "LOW_BATTERY": "🔋 Low Battery Warning (Land immediately)",
        "BATTERY": "🔋 Battery Failsafe Triggered",
        "EKF": "🧭 EKF Variance (GPS/Compass interference)",
        "VIBE": "🫨 High Vibration Level (Check props/mounting)",
        "RC": "📡 RC Failsafe (Signal Lost)",
        "FAILSAFE": "⚠️ Failsafe Triggered (RTL or Land)",
        "BARO": "☁️ Barometer Error (Altitude drift risk)",
        "GPS": "🛰️ GPS Glitch or Loss",
        "ARMING": "⚙️ Arming Check / Status",
        "ERR": "❗ System Error",
    }
    for key, desc in translations.items():
        if key in full_text:
            return desc
    return "ℹ️ Standard Event Log"


def build_detailed_stats_string(data):
    df_bat = data.get("df_bat")
    df_vibe = data.get("df_vibe")
    err_messages = data.get("err_messages", [])

    # Flight time (minutes)
    flight_minutes = "N/A"
    duration_seconds = data.get("duration_seconds")
    if duration_seconds is not None:
        try:
            flight_minutes = f"{(float(duration_seconds) / 60):.2f} min"
        except (TypeError, ValueError):
            flight_minutes = "N/A"
    elif df_bat is not None and len(df_bat) > 0:
        time_max = df_bat["TimeS"].max() if "TimeS" in df_bat.columns else df_bat["time_normalized"].max()
        flight_minutes = f"{(time_max / 60):.2f} min"
    elif df_vibe is not None and len(df_vibe) > 0:
        time_max = df_vibe["TimeS"].max() if "TimeS" in df_vibe.columns else df_vibe["time_normalized"].max()
        flight_minutes = f"{(time_max / 60):.2f} min"

    # Battery health
    start_v = end_v = min_v = total_ah = "N/A"
    if df_bat is not None and len(df_bat) > 0:
        start_v = f"{float(df_bat['volt'].iloc[0]):.2f} V"
        end_v = f"{float(df_bat['volt'].iloc[-1]):.2f} V"
        min_v = f"{float(df_bat['volt'].min()):.2f} V"
        if "TimeS" in df_bat.columns:
            dt = df_bat["TimeS"].diff().fillna(0).clip(lower=0)
        elif "time_normalized" in df_bat.columns:
            dt = df_bat["time_normalized"].diff().fillna(0).clip(lower=0)
            total_ah_val = float((df_bat["curr"] * dt).sum() / 3600)
            total_ah = f"{total_ah_val:.3f} Ah"

    # Vibration analysis
    vibe_stats = "N/A"
    if df_vibe is not None and len(df_vibe) > 0:
        vx_mean = df_vibe["x"].mean()
        vy_mean = df_vibe["y"].mean()
        vz_mean = df_vibe["z"].mean()
        vx_max = df_vibe["x"].max()
        vy_max = df_vibe["y"].max()
        vz_max = df_vibe["z"].max()
        clipping = any(val > 30 for val in [vx_max, vy_max, vz_max])
        vibe_stats = (
            f"VibeX mean/max: {vx_mean:.2f}/{vx_max:.2f}, "
            f"VibeY mean/max: {vy_mean:.2f}/{vy_max:.2f}, "
            f"VibeZ mean/max: {vz_mean:.2f}/{vz_max:.2f}, "
            f"Clipping: {'YES' if clipping else 'NO'}"
        )

    # Motor balance (PWM)
    motor_balance = "N/A (PWM data not available)"

    # GPS health
    gps_health = "N/A (HDOP/Sat count not available)"

    # Errors
    if err_messages:
        errors_text = "\n".join([f"- {msg}" for msg in err_messages])
    else:
        errors_text = "None"

    detailed_stats = f"""
Flight Time: {flight_minutes}
Battery Health:
  - Start Voltage: {start_v}
  - End Voltage: {end_v}
  - Voltage Sag (Min): {min_v}
  - Total Current Consumed: {total_ah}
Vibration Analysis:
  - {vibe_stats}
Motor Balance:
  - {motor_balance}
GPS Health:
  - {gps_health}
Errors:
{errors_text}
""".strip()

    return detailed_stats


def create_pdf_report(log_summary, ai_analysis, filename):
    """
    Generates a PDF report in English only.
    Non-English characters are stripped out to prevent errors.
    Returns: (bytes, error_message)
    """
    try:
        # 1. Text Sanitizer (The Magic Fix)
        def clean_text(text):
            if not text:
                return ""
            # 1. Convert to string
            text = str(text)
            # 2. Remove Markdown symbols (*, #, `) that look messy in plain text
            text = text.replace('*', '').replace('#', '').replace('`', '')
            # 3. Force English (Latin-1), ignore non-English chars
            return text.encode('latin-1', 'ignore').decode('latin-1')

        pdf = FPDF()
        pdf.add_page()

        # Layout settings
        margin = 10
        pdf.set_left_margin(margin)
        pdf.set_right_margin(margin)
        epw = pdf.w - (margin * 2)

        # 2. Header (English)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(epw, 10, "Drone Log Analysis Report", ln=True, align='C')
        pdf.ln(10)

        # 3. Flight Overview (English)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(epw, 10, "1. Flight Overview", ln=True)

        pdf.set_font("Arial", "", 10)
        for key, value in log_summary.items():
            # Clean both key and value to ensure no Korean slips in
            safe_line = f"{clean_text(key)}: {clean_text(str(value))}"
            pdf.multi_cell(epw, 7, safe_line)

        pdf.ln(5)

        # 4. AI Diagnosis (English Content Only)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(epw, 10, "2. AI Diagnosis Details", ln=True)

        pdf.set_font("Arial", "", 10)
        # If AI analysis was in Korean, this might become empty or just show English terms.
        # This is expected behavior to avoid crashes/question marks.
        pdf.multi_cell(epw, 7, clean_text(ai_analysis))

        # 5. Footer
        pdf.ln(20)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(epw, 10, "Generated by AI Drone Log Analyzer", align='C')

        # 6. Return Bytes (Streamlit compatible)
        return bytes(pdf.output()), None

    except Exception as e:
        return None, str(e)


def call_openai_api(messages, api_key):
    """
    Call OpenAI API with the conversation messages.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def _parse_message_stream(master, max_none=MAX_CONSECUTIVE_NONE):
    """Generic message stream parser with consecutive None handling."""
    consecutive_none = 0
    while True:
        msg = master.recv_match(blocking=False)
        if msg is None:
            consecutive_none += 1
            if consecutive_none >= max_none:
                break
            continue
        consecutive_none = 0
        yield msg

def _extract_gps_coords(msg):
    """Extract GPS coordinates from message."""
    lat = _safe_getattr(msg, "Lat", "lat")
    lon = _safe_getattr(msg, "Lng", "lon", "Lon")
    if lat is None or lon is None:
        return None
    if abs(lat) > 90 or abs(lon) > 180:
        lat, lon = lat / 1e7, lon / 1e7
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return (lat, lon)
    return None

def _calculate_arm_duration(arm_events, fallback_end_us=None):
    """Calculate total armed duration in seconds from EV arm/disarm events."""
    if not arm_events:
        return None
    events_sorted = sorted(arm_events, key=lambda item: item[0])
    total_us = 0.0
    arm_start = None
    for time_us, ev_value in events_sorted:
        if ev_value == 10:
            if arm_start is None:
                arm_start = time_us
        elif ev_value == 11:
            if arm_start is not None and time_us >= arm_start:
                total_us += time_us - arm_start
                arm_start = None
    if arm_start is not None:
        end_us = fallback_end_us if fallback_end_us is not None else events_sorted[-1][0]
        if end_us >= arm_start:
            total_us += end_us - arm_start
    if total_us <= 0:
        return None
    return total_us / 1e6

@st.cache_data(show_spinner=False)
def analyze_log_file(file_bytes: bytes):
    """Parse the log file once and cache results."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    tfile.write(file_bytes)
    tfile.close()
    file_path = os.path.abspath(tfile.name)
    
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            first_bytes = f.read(16)
        
        # First pass: message types + validity flags
        master = mavutil.mavlink_connection(file_path)
        message_types = set()
        message_count = 0
        has_parm = False
        has_msg = False
        err_messages = []
        ev_messages = []
        arm_events = []
        gps_time_us = []
        parm_time_us = []
        
        for msg in _parse_message_stream(master):
            message_count += 1
            msg_type = msg.get_type()
            message_types.add(msg_type)

            if msg_type == "PARM":
                has_parm = True
                parm_time = _extract_time_us(msg)
                if parm_time is not None:
                    parm_time_us.append(parm_time)
            elif msg_type == "MSG":
                has_msg = True
            elif msg_type == "ERR" and len(err_messages) < 50:
                err_messages.append(str(msg))
            elif msg_type == "EV" and len(ev_messages) < 50:
                ev_messages.append(str(msg))
                ev_value = _safe_getattr(msg, "Id", "id", "Event", "event", "EV", "E", "Value", "value")
                ev_time = _extract_time_us(msg)
                try:
                    ev_value_int = int(ev_value) if ev_value is not None else None
                except (TypeError, ValueError):
                    ev_value_int = None
                if ev_value_int in (10, 11) and ev_time is not None:
                    arm_events.append((ev_time, ev_value_int))
            elif msg_type in GPS_MESSAGE_TYPES:
                gps_time = _extract_time_us(msg)
                if gps_time is not None:
                    gps_time_us.append(gps_time)
        
        master.close()

        duration_sec = 0
        fallback_end_us = None
        if gps_time_us:
            fallback_end_us = max(gps_time_us)
        elif parm_time_us:
            fallback_end_us = max(parm_time_us)

        arm_duration = _calculate_arm_duration(arm_events, fallback_end_us=fallback_end_us)
        if arm_duration is not None:
            duration_sec = arm_duration
        elif gps_time_us:
            duration_sec = (max(gps_time_us) - min(gps_time_us)) / 1e6
        elif parm_time_us:
            duration_sec = (max(parm_time_us) - min(parm_time_us)) / 1e6
        
        # Second pass: GPS track
        gps_points = []
        try:
            master_gps = mavutil.mavlink_connection(file_path)
            for msg in _parse_message_stream(master_gps, max_none=200):
                if msg.get_type() in GPS_MESSAGE_TYPES:
                    coords = _extract_gps_coords(msg)
                    if coords:
                        gps_points.append(coords)
        except Exception:
            pass
        
        # Third pass: VIBE data
        vibe_data = []
        if 'VIBE' in message_types:
            try:
                master_vibe = mavutil.mavlink_connection(file_path)
                vibe_count = 0
                for msg in _parse_message_stream(master_vibe, max_none=100):
                    if msg.get_type() == 'VIBE':
                        vibe_count += 1
                        vibe_x = _safe_getattr(msg, 'VibeX', 'vibe_x', 'x')
                        vibe_y = _safe_getattr(msg, 'VibeY', 'vibe_y', 'y')
                        vibe_z = _safe_getattr(msg, 'VibeZ', 'vibe_z', 'z')
                        timestamp_us = _extract_time_us(msg)
                        if timestamp_us is None:
                            timestamp_us = float(vibe_count) * 1e6

                        if vibe_x is not None or vibe_y is not None or vibe_z is not None:
                            vibe_data.append({
                                'TimeUS': timestamp_us,
                                'x': float(vibe_x) if vibe_x is not None else 0,
                                'y': float(vibe_y) if vibe_y is not None else 0,
                                'z': float(vibe_z) if vibe_z is not None else 0
                            })
            except Exception:
                pass

        # Fourth pass: BAT data
        bat_data = []
        if 'BAT' in message_types:
            try:
                master_bat = mavutil.mavlink_connection(file_path)
                bat_count = 0
                for msg in _parse_message_stream(master_bat, max_none=100):
                    if msg.get_type() == 'BAT':
                        bat_count += 1
                        volt = _safe_getattr(msg, 'Volt', 'volt', 'V')
                        curr = _safe_getattr(msg, 'Curr', 'curr', 'I')
                        timestamp_us = _extract_time_us(msg)
                        if timestamp_us is None:
                            timestamp_us = float(bat_count) * 1e6

                        if volt is not None or curr is not None:
                            bat_data.append({
                                'TimeUS': timestamp_us,
                                'volt': float(volt) if volt is not None else 0.0,
                                'curr': float(curr) if curr is not None else 0.0,
                            })
            except Exception:
                pass
        
        # Process DataFrames
        df_vibe = None
        if vibe_data:
            df_vibe = pd.DataFrame(vibe_data)
            if 'TimeUS' in df_vibe.columns:
                df_vibe['TimeS'] = (df_vibe['TimeUS'] - df_vibe['TimeUS'].min()) / 1e6
                df_vibe['time_normalized'] = df_vibe['TimeS']
            else:
                df_vibe['TimeS'] = range(len(df_vibe))
                df_vibe['time_normalized'] = df_vibe['TimeS']

        df_bat = None
        if bat_data:
            df_bat = pd.DataFrame(bat_data)
            if 'TimeUS' in df_bat.columns:
                df_bat['TimeS'] = (df_bat['TimeUS'] - df_bat['TimeUS'].min()) / 1e6
                df_bat['time_normalized'] = df_bat['TimeS']
            else:
                df_bat['TimeS'] = range(len(df_bat))
                df_bat['time_normalized'] = df_bat['TimeS']
        
        return {
            'file_size': file_size,
            'first_bytes': first_bytes,
            'message_types': sorted(message_types),
            'message_count': message_count,
            'has_parm': has_parm,
            'has_msg': has_msg,
            'gps_points': gps_points,
            'df_vibe': df_vibe,
            'df_bat': df_bat,
            'duration_seconds': duration_sec,
            'log_summary': generate_log_summary(file_path),
            'err_messages': err_messages,
            'ev_messages': ev_messages,
        }
    
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass


# 2. File upload
uploaded_file = st.file_uploader("Select a log file (.bin)", type=["bin"])

if uploaded_file is not None:
    # 3. When a file is uploaded
    st.success("File uploaded successfully. Starting analysis...")
    
    # Get file bytes and analyze (cached - only runs once per unique file)
    file_bytes = uploaded_file.getvalue()
    
    with st.spinner("Parsing log file... This may take a moment."):
        data = analyze_log_file(file_bytes)
    
    # Extract cached results
    file_size = data['file_size']
    first_bytes = data['first_bytes']
    message_types = data['message_types']
    message_count = data['message_count']
    has_parm = data['has_parm']
    has_msg = data['has_msg']
    gps_points = data['gps_points']
    df_vibe = data['df_vibe']
    df_bat = data['df_bat']
    err_messages = data.get('err_messages', [])
    ev_messages = data.get('ev_messages', [])
    
    # Display file info
    st.write(f"**File size:** {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    st.write(f"**File header bytes:** {first_bytes.hex()}")
    
    # High‑level log info
    st.subheader("📂 Basic log information")
    st.write(f"**Total parsed messages:** {message_count:,}")
    
    if message_types:
        st.success(f"✅ Log parsed successfully (message types found: {len(message_types)})")
        visible_limit = 30
        visible_types = message_types[:visible_limit]
        extra_count = len(message_types) - visible_limit
        msg_text = ", ".join(visible_types)
        if extra_count > 0:
            st.write(f"**Message types:** {msg_text} ... and {extra_count} more")
        else:
            st.write(f"**Message types:** {msg_text}")
    else:
        st.error("❌ No messages could be parsed from this file.")
        st.write("**Possible reasons:**")
        st.write("- The file is empty or corrupted")
        st.write("- The file is not an Ardupilot DataFlash log")
        st.write("- The file is encrypted or in an unexpected format")
        st.write(f"- File header bytes: `{first_bytes.hex()}`")
    
    # Simple validity hint based on PARM / MSG presence
    if has_parm or has_msg:
        st.info("Log contains configuration/MSG records (PARM/MSG), which is typical for Ardupilot DataFlash logs.")
    elif message_count > 0:
        st.warning("No PARM or MSG records were found. The file may still be valid, but this is less common.")
    
    st.markdown("### Flight Analysis Dashboard")
    tab_map, tab_charts, tab_chat = st.tabs(["🗺️ Map", "📊 Charts", "🤖 AI Chat"])

    with tab_charts:
        st.write("#### 🔋 Battery & Power")
        if df_bat is not None and len(df_bat) > 0:
            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax2 = ax1.twinx()

            # Voltage on left Y-axis
            ax1.plot(
                df_bat['TimeS'],
                df_bat['volt'],
                color='tab:blue',
                linewidth=0.7,
                alpha=0.9,
                label='Voltage (V)',
            )
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Voltage (V)", color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            # Current on right Y-axis
            ax2.plot(
                df_bat['TimeS'],
                df_bat['curr'],
                color='tab:red',
                linewidth=0.7,
                alpha=0.9,
                label='Current (A)',
            )
            ax2.set_ylabel("Current (A)", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            ax1.grid(True, alpha=0.3)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.tight_layout()
            st.pyplot(fig)

            # Summary metrics
            min_volt = float(df_bat['volt'].min())
            max_curr = float(df_bat['curr'].max())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Min Voltage (V)", f"{min_volt:.2f}")
            with col2:
                st.metric("Max Current (A)", f"{max_curr:.2f}")
        else:
            st.info("No BAT data available for charting.")

        st.write("#### 🔍 Vibration")
        if df_vibe is not None and len(df_vibe) > 0:
            st.write(f"**VIBE messages collected:** {len(df_vibe):,}")

            # Plot vibrations
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle('Vibration Analysis', fontsize=16, fontweight='bold')

            # X‑axis vibration
            axes[0].plot(df_vibe['TimeS'], df_vibe['x'], 'r-', linewidth=0.5, alpha=0.7)
            axes[0].set_title('Vibration X', fontsize=12)
            axes[0].set_ylabel('Value', fontsize=10)
            axes[0].grid(True, alpha=0.3)

            # Y‑axis vibration
            axes[1].plot(df_vibe['TimeS'], df_vibe['y'], 'g-', linewidth=0.5, alpha=0.7)
            axes[1].set_title('Vibration Y', fontsize=12)
            axes[1].set_ylabel('Value', fontsize=10)
            axes[1].grid(True, alpha=0.3)

            # Z‑axis vibration
            axes[2].plot(df_vibe['TimeS'], df_vibe['z'], 'b-', linewidth=0.5, alpha=0.7)
            axes[2].set_title('Vibration Z', fontsize=12)
            axes[2].set_xlabel('Time (seconds)', fontsize=10)
            axes[2].set_ylabel('Value', fontsize=10)
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            for i, axis in enumerate(['x', 'y', 'z']):
                with [col1, col2, col3][i]:
                    st.metric(f"{axis.upper()} mean", f"{df_vibe[axis].mean():.2f}")
                    st.metric(f"{axis.upper()} max", f"{df_vibe[axis].max():.2f}")
                    st.metric(f"{axis.upper()} min", f"{df_vibe[axis].min():.2f}")
        elif 'VIBE' in message_types:
            st.warning("VIBE messages were detected, but no usable vibration data could be extracted.")
        else:
            st.info("No VIBE data available for charting.")

        st.write("#### 🧯 Error & Event analysis")
        if err_messages or ev_messages:
            rows = []
            for msg in err_messages:
                rows.append({
                    "Time": "N/A",
                    "Type": "ERR",
                    "Message": msg,
                    "💡 Explanation": get_error_desc("ERR", msg),
                })
            for msg in ev_messages:
                rows.append({
                    "Time": "N/A",
                    "Type": "EV",
                    "Message": msg,
                    "💡 Explanation": get_error_desc("EV", msg),
                })
            df_errors = pd.DataFrame(rows)
            st.dataframe(
                df_errors[["Time", "Type", "Message", "💡 Explanation"]],
                use_container_width=True,
            )
        else:
            st.info("No ERR/EV records found in this log.")

    with tab_map:
        st.write("#### 🗺️ 3D Flight Path (Google Hybrid)")

        map_data = pd.DataFrame(gps_points, columns=["lat", "lon"])
        if not map_data.empty and "lat" in map_data.columns and "lon" in map_data.columns:
            # 1. OPTIMIZATION: Downsample to prevent crash
            display_data = map_data
            if len(map_data) > 500:
                display_data = map_data.iloc[::10, :]
                st.caption(f"ℹ️ Map optimized: Displaying {len(display_data)} points.")

            # 2. AUTO-CENTER
            mid_lat = display_data["lat"].mean()
            mid_lon = display_data["lon"].mean()

            view_state = pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=17,
                pitch=45,
                bearing=0,
            )

            # 3. BACKGROUND: Google Hybrid (Satellite + Roads)
            google_hybrid_layer = pdk.Layer(
                "TileLayer",
                data=["https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"],
                get_line_color=[0, 0, 0],
                max_zoom=20,
                opacity=1.0,
            )

            # 4. DATA: Red Flight Path (Scatterplot)
            path_layer = pdk.Layer(
                "ScatterplotLayer",
                display_data,
                get_position=["lon", "lat"],
                get_color=[255, 0, 0, 200],
                get_radius=2,
                pickable=True,
            )

            # 5. RENDER
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[google_hybrid_layer, path_layer],
                tooltip={"text": "Lat: {lat}\nLon: {lon}"},
            ))
        else:
            st.warning("⚠️ No GPS data found in this log.")

    # Store analyzed data in session state for chatbot access
    st.session_state.analyzed_data = data

    with tab_chat:
        st.subheader("🤖 GPT-5.2 Copilot")

        # 1. Check if Client exists in Session State
        if "openai_client" not in st.session_state:
            st.warning("⚠️ 코드를 열어서 'MY_API_KEY' 부분에 sk-로 시작하는 OpenAI 키를 입력해주세요!")
        else:
            client = st.session_state.openai_client

            # 2. Initialize History
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # 3. Display History
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("pdf_data"):
                        st.download_button(
                            label="📄 Download Report (PDF)",
                            data=message["pdf_data"],
                            file_name="drone_analysis_report.pdf",
                            mime="application/pdf",
                            key=f"history_btn_{i}"
                        )

            # 4. Chat Input & Response
            if prompt := st.chat_input("Ask about your drone log..."):
                
                # Show User Message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate AI Response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 GPT-5.2 is thinking..."):
                        try:
                            log_context = "No log data."
                            if "analyzed_data" in st.session_state:
                                log_context = build_detailed_stats_string(st.session_state.analyzed_data)
                            safe_context = _safe_unicode_text(log_context)
                            messages_payload = [
                                {"role": "system", "content": f"You are a Senior ArduPilot log analyst. The log file is already uploaded and parsed. Do not ask the user to provide the .bin file. Provide a detailed, expert-level analysis with clear technical reasoning, cite relevant metrics from the data, explain likely causes, and give prioritized actionable recommendations. Use the user's language. Analyze this data: {safe_context}"},
                                *[
                                    {"role": m["role"], "content": _safe_unicode_text(m["content"])}
                                    for m in st.session_state.messages
                                ],
                            ]
                            completion = client.chat.completions.create(
                                model="gpt-5.2",
                                messages=messages_payload,
                            )
                            response_text = completion.choices[0].message.content

                            st.markdown(response_text)

                            # PDF & Save
                            pdf_bytes = None
                            create_pdf_func = globals().get("create_pdf")
                            if create_pdf_func:
                                pdf_bytes = create_pdf_func(response_text)
                                st.download_button(
                                    label="📄 Download Report (PDF)",
                                    data=pdf_bytes,
                                    file_name="report.pdf",
                                    mime="application/pdf",
                                    key=f"new_{len(st.session_state.messages)}"
                                )

                            msg_data = {"role": "assistant", "content": response_text}
                            if pdf_bytes:
                                msg_data["pdf_data"] = pdf_bytes
                            st.session_state.messages.append(msg_data)

                            st.components.v1.html(
                                """<script>
                                    setTimeout(function(){
                                        var v = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                                        if (v) v.scrollTop = v.scrollHeight;
                                    }, 50);
                                </script>""",
                                height=0,
                            )
                        except UnicodeEncodeError as e:
                            st.error(f"🚨 GPT Error: UTF-8 인코딩 실패 - {e}")
                        except Exception as e:
                            st.error(f"🚨 GPT Error: {e}")
