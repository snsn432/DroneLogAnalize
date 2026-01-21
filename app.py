import streamlit as st
import pandas as pd
from pymavlink import mavutil
import tempfile
import os
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from openai import OpenAI

# Font settings
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Constants
GPS_MESSAGE_TYPES = ("GPS", "GPS2", "GPS_RAW", "GPS_RAW_INT", "GPS2_RAW")
MAX_CONSECUTIVE_NONE = 500

# 1. Page layout & header
st.set_page_config(page_title="Ardupilot Log Analyzer", layout="wide")

# --- HIDE STREAMLIT UI ELEMENTS ---
hide_st_style = """
<style>
    /* 1. 헤더(Header) 전체 틀은 보이게 유지 (사이드바 버튼 때문) */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* 2. 오른쪽 상단 버튼 그룹 (Fork, GitHub, 점3개) 숨기기 */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* 3. 오른쪽 하단 뱃지/블럭 (ViewerBadge) 숨기기 */
    div[class*="viewerBadge"] {
        display: none !important;
    }

    /* 4. 하단 푸터 (Made with Streamlit) 숨기기 */
    footer {
        display: none !important;
    }
    
    /* 5. 우측 상단 배포 버튼 숨기기 */
    .stDeployButton {
        display: none !important;
    }
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🚁 AI‑based Ardupilot Log Analyzer")
st.write("Upload an Ardupilot log file (.bin) to inspect basic info and vibration data.")

# Pro license state
if "is_pro" not in st.session_state:
    st.session_state.is_pro = False

# Sidebar: Pro license login
with st.sidebar:
    st.header("🔑 Pro License Key")
    pro_key = st.text_input(
        "Pro License Key",
        type="password",
        help="Enter your Pro license key to unlock advanced features.",
    )

    pro_password = st.secrets.get("PRO_PASSWORD", None)
    if pro_key and pro_password and pro_key == pro_password:
        st.session_state.is_pro = True
        st.success("✅ Pro features unlocked!")
    else:
        st.session_state.is_pro = False
        st.info("🔒 Advanced features are locked.")
        st.link_button(
            "🚀 Get Lifetime Access ($9)",
            "https://scownu.gumroad.com/l/fdyrdb",
        )

def _safe_getattr(msg, *attrs):
    """Safely get attribute from message with multiple fallback names."""
    for attr in attrs:
        val = getattr(msg, attr, None)
        if val is not None:
            return val
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
        
        for msg in _parse_message_stream(master):
            message_count += 1
            msg_type = msg.get_type()
            message_types.add(msg_type)
            
            if msg_type == "PARM":
                has_parm = True
            elif msg_type == "MSG":
                has_msg = True
            elif msg_type == "ERR" and len(err_messages) < 50:
                err_messages.append(str(msg))
            elif msg_type == "EV" and len(ev_messages) < 50:
                ev_messages.append(str(msg))
        
        master.close()
        
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
                        timestamp = _safe_getattr(msg, 'time_usec', 'time_boot_ms') or vibe_count
                        
                        if vibe_x is not None or vibe_y is not None or vibe_z is not None:
                            vibe_data.append({
                                'time': timestamp,
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
                        timestamp = _safe_getattr(msg, 'time_usec', 'time_boot_ms') or bat_count
                        
                        if volt is not None or curr is not None:
                            bat_data.append({
                                'time': timestamp,
                                'volt': float(volt) if volt is not None else 0.0,
                                'curr': float(curr) if curr is not None else 0.0,
                            })
            except Exception:
                pass
        
        # Process DataFrames
        df_vibe = None
        if vibe_data:
            df_vibe = pd.DataFrame(vibe_data)
            time_col = df_vibe['time']
            df_vibe['time_normalized'] = (time_col - time_col.min()) / 1000000 if 'time' in df_vibe.columns else range(len(df_vibe))

        df_bat = None
        if bat_data:
            df_bat = pd.DataFrame(bat_data)
            time_col = df_bat['time']
            df_bat['time_normalized'] = (time_col - time_col.min()) / 1000000 if 'time' in df_bat.columns else range(len(df_bat))
        
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
    
    # GPS track: draw flight path on a map using folium (Pro only)
    st.write("---")
    st.subheader("🗺️ Flight path (GPS)")
    
    if st.session_state.is_pro:
        if len(gps_points) >= 2:
            start_lat, start_lon = gps_points[0]
            end_lat, end_lon = gps_points[-1]
            
            # Use Esri World Imagery (satellite) tiles and allow deep zoom
            fmap = folium.Map(
                location=[start_lat, start_lon],
                zoom_start=16,
                tiles=None,
                max_zoom=20,
            )
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="Esri World Imagery",
                max_zoom=20,
            ).add_to(fmap)

            # Draw flight path
            folium.PolyLine(gps_points, color="red", weight=3, opacity=0.8).add_to(fmap)
            folium.Marker(
                location=[start_lat, start_lon],
                tooltip="Start",
                icon=folium.Icon(color="green", icon="play", prefix="fa"),
            ).add_to(fmap)
            folium.Marker(
                location=[end_lat, end_lon],
                tooltip="End",
                icon=folium.Icon(color="red", icon="flag", prefix="fa"),
            ).add_to(fmap)

            # Automatically fit map bounds to the whole flight path
            fmap.fit_bounds(gps_points)
            
            st_folium(fmap, key="flight_map", width="100%", height=500)
        else:
            st.info("No usable GPS track could be extracted from this log.")
    else:
        st.info("🔒 This feature is Pro-only.")

    # Error & Event analysis (Pro only)
    st.write("---")
    st.subheader("🧯 Error & Event analysis")
    if st.session_state.is_pro:
        if err_messages or ev_messages:
            rows = []
            for msg in err_messages:
                rows.append({"Type": "ERR", "Message": msg})
            for msg in ev_messages:
                rows.append({"Type": "EV", "Message": msg})
            df_errors = pd.DataFrame(rows)
            st.dataframe(df_errors, use_container_width=True)
        else:
            st.info("No ERR/EV records found in this log.")
    else:
        st.info("🔒 This feature is Pro-only.")

    # Battery analysis (BAT)
    if df_bat is not None and len(df_bat) > 0:
        st.write("---")
        st.subheader("🔋 Battery analysis (BAT)")

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()

        # Voltage on left Y-axis
        ax1.plot(
            df_bat['time_normalized'],
            df_bat['volt'],
            color='tab:blue',
            linewidth=0.7,
            alpha=0.9,
            label='Voltage (V)',
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Voltage (V)", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Current on right Y-axis
        ax2.plot(
            df_bat['time_normalized'],
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
    
    # VIBE vibration analysis
    if df_vibe is not None and len(df_vibe) > 0:
        st.write("---")
        st.write("**🔍 Vibration analysis (VIBE messages)**")
        st.write(f"**VIBE messages collected:** {len(df_vibe):,}")
        
        # Plot vibrations
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Vibration Analysis', fontsize=16, fontweight='bold')
        
        # X‑axis vibration
        axes[0].plot(df_vibe['time_normalized'], df_vibe['x'], 'r-', linewidth=0.5, alpha=0.7)
        axes[0].set_title('Vibration X', fontsize=12)
        axes[0].set_ylabel('Value', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Y‑axis vibration
        axes[1].plot(df_vibe['time_normalized'], df_vibe['y'], 'g-', linewidth=0.5, alpha=0.7)
        axes[1].set_title('Vibration Y', fontsize=12)
        axes[1].set_ylabel('Value', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Z‑axis vibration
        axes[2].plot(df_vibe['time_normalized'], df_vibe['z'], 'b-', linewidth=0.5, alpha=0.7)
        axes[2].set_title('Vibration Z', fontsize=12)
        axes[2].set_xlabel('Time (s)', fontsize=10)
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
        st.write("---")
        st.write("**🔍 Vibration analysis (VIBE messages)**")
        st.warning("VIBE messages were detected, but no usable vibration data could be extracted.")
    
    # Store analyzed data in session state for chatbot access
    st.session_state.analyzed_data = data

# AI Chatbot section (Pro only)
st.write("---")
st.subheader("🤖 AI Drone Consultation Chatbot")

if not st.session_state.is_pro:
    st.info("🔒 This feature is Pro-only.")
else:
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Please contact the administrator (API key configuration error).")
    else:
        # Generate log summary for context injection (if data exists)
        if "analyzed_data" in st.session_state:
            log_summary = st.session_state.analyzed_data.get("log_summary", "")
            system_prompt = f"""You are a world-class Ardupilot Log Analysis Expert with 25+ years of experience in autonomous flight systems, MAVLink protocol analysis, and drone diagnostics. You specialize in identifying flight anomalies, diagnosing hardware issues, and providing actionable maintenance recommendations.

**Your Expertise Includes:**
- Deep understanding of Ardupilot firmware, flight modes, and control algorithms
- Comprehensive knowledge of MAVLink message types (ATT, GPS, VIBE, BAT, CTUN, ERR, EV, etc.)
- Advanced interpretation of telemetry data, sensor readings, and flight dynamics
- Root cause analysis of vibration patterns, GPS issues, battery degradation, and attitude control problems
- Industry best practices for drone maintenance, calibration, and troubleshooting

**Analysis Framework:**
1. **Data Interpretation**: Analyze all provided metrics (ATT errors, GPS quality, vibration levels, battery health, throttle outputs) in context
2. **Pattern Recognition**: Identify correlations between different message types and flight phases
3. **Anomaly Detection**: Flag unusual patterns, outliers, or concerning trends that may indicate issues
4. **Root Cause Analysis**: Trace symptoms back to potential hardware/software/firmware causes
5. **Actionable Recommendations**: Provide specific, prioritized steps for investigation and resolution

**Flight Log Summary:**
{log_summary}

**Your Response Style:**
- Answer in English only
- Be professional, thorough, and technically precise
- Use clear structure: Observation → Analysis → Recommendation
- Quantify issues when possible (e.g., "VIBE X-axis shows 15% above normal threshold")
- Prioritize safety-critical issues first
- If data is insufficient, clearly state assumptions and recommend additional diagnostics
- Reference specific message types and values from the log when relevant

Answer the user's questions based on this flight log summary. For questions beyond the log data, provide expert guidance on drone operations, maintenance protocols, and industry best practices while clearly distinguishing between log-based analysis and general recommendations."""
        else:
            system_prompt = """You are a world-class Ardupilot Log Analysis Expert with 25+ years of experience in autonomous flight systems, MAVLink protocol analysis, and drone diagnostics.

**Your Expertise Includes:**
- Deep understanding of Ardupilot firmware, flight modes, and control algorithms
- Comprehensive knowledge of MAVLink message types and their significance
- Advanced interpretation of telemetry data, sensor readings, and flight dynamics
- Root cause analysis of common drone issues
- Industry best practices for drone maintenance, calibration, and troubleshooting

**When No Log is Available:**
Provide comprehensive guidance on:
1. **Log Analysis Fundamentals**: How to interpret key message types (ATT, GPS, VIBE, BAT, CTUN, ERR, EV)
2. **Common Anomalies**: Typical issues seen in logs (GPS glitches, vibration problems, battery degradation, attitude control errors)
3. **Metric Interpretation**: 
   - Vibration thresholds and what high values indicate
   - Battery voltage/current patterns and health indicators
   - GPS quality metrics (NSats, HDOP) and their impact on flight safety
   - Attitude control errors and their relationship to flight stability
4. **Maintenance Best Practices**: Calibration procedures, preventive maintenance schedules, and diagnostic workflows
5. **Troubleshooting Workflows**: Systematic approaches to identifying and resolving issues

**Your Response Style:**
- Answer in English only
- Be professional, thorough, and technically precise
- Use clear structure and examples
- Reference industry standards and best practices
- Provide actionable guidance that users can implement"""

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about your drone log data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Prepare messages for API call (system + conversation history)
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend({"role": msg["role"], "content": msg["content"]} 
                               for msg in st.session_state.messages[-10:])
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = call_openai_api(api_messages, api_key)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
