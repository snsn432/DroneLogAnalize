import streamlit as st
import pandas as pd
from pymavlink import mavutil
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import folium
from streamlit_folium import st_folium
from openai import OpenAI

# Font settings (keep UTF-8 & minus sign clean)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Works fine for English + Korean on Windows
plt.rcParams['axes.unicode_minus'] = False

# 1. Page layout & header
st.set_page_config(page_title="Ardupilot Log Analyzer", layout="wide")

# Hide Streamlit default UI elements + mobile layout tweaks
st.markdown("""
    <style>
    #MainMenu {display: none !important;}
    header {display: none !important;}
    footer {display: none !important;}
    .stDeployButton {display:none !important;}
    [data-testid="stStatusWidget"] {display: none !important;}
    .viewerBadge_container__1QS1n {display: none !important;}

    /* Mobile spacing adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

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

def generate_log_summary(file_path: str) -> str:
    """
    Parse the log with pymavlink and return a concise summary string
    for AI context injection.
    """
    from pymavlink import mavutil

    # ATT (attitude)
    att_roll_errors = []
    att_pitch_errors = []

    # GPS (precision)
    nsats_min = None
    hdop_max = None

    # ERR / EV
    err_messages = []
    ev_messages = []

    # CTUN (throttle out)
    ctnu_tho_values = []

    # VIBE
    vibe_x_vals = []
    vibe_y_vals = []
    vibe_z_vals = []

    # BAT
    bat_voltages = []

    master = mavutil.mavlink_connection(file_path)
    consecutive_none = 0

    while True:
        msg = master.recv_match(blocking=False)
        if msg is None:
            consecutive_none += 1
            if consecutive_none >= 500:
                break
            continue

        consecutive_none = 0
        mtype = msg.get_type()

        # ATT: Roll/Pitch errors
        if mtype == "ATT":
            des_roll = getattr(msg, "DesRoll", getattr(msg, "desroll", None))
            roll = getattr(msg, "Roll", getattr(msg, "roll", None))
            des_pitch = getattr(msg, "DesPitch", getattr(msg, "despitch", None))
            pitch = getattr(msg, "Pitch", getattr(msg, "pitch", None))
            if des_roll is not None and roll is not None:
                att_roll_errors.append(abs(float(des_roll) - float(roll)))
            if des_pitch is not None and pitch is not None:
                att_pitch_errors.append(abs(float(des_pitch) - float(pitch)))

        # GPS: NSats min, HDOP max
        if mtype in ("GPS", "GPS2", "GPS_RAW", "GPS_RAW_INT", "GPS2_RAW"):
            nsats = getattr(msg, "NSats", getattr(msg, "nsats", None))
            hdop = getattr(msg, "HDop", getattr(msg, "HDOP", getattr(msg, "hdop", None)))
            if nsats is not None:
                nsats = int(nsats)
                nsats_min = nsats if nsats_min is None else min(nsats_min, nsats)
            if hdop is not None:
                hdop = float(hdop)
                hdop_max = hdop if hdop_max is None else max(hdop_max, hdop)

        # ERR / EV messages (limit later)
        if mtype == "ERR":
            err_messages.append(str(msg))
        elif mtype == "EV":
            ev_messages.append(str(msg))

        # CTUN: ThO
        if mtype == "CTUN":
            tho = getattr(msg, "ThO", getattr(msg, "tho", getattr(msg, "THO", None)))
            if tho is not None:
                ctnu_tho_values.append(float(tho))

        # VIBE: VibeX/Y/Z
        if mtype == "VIBE":
            vx = getattr(msg, "VibeX", getattr(msg, "vibe_x", None))
            vy = getattr(msg, "VibeY", getattr(msg, "vibe_y", None))
            vz = getattr(msg, "VibeZ", getattr(msg, "vibe_z", None))
            if vx is not None:
                vibe_x_vals.append(float(vx))
            if vy is not None:
                vibe_y_vals.append(float(vy))
            if vz is not None:
                vibe_z_vals.append(float(vz))

        # BAT: Volt
        if mtype == "BAT":
            volt = getattr(msg, "Volt", getattr(msg, "volt", getattr(msg, "V", None)))
            if volt is not None:
                bat_voltages.append(float(volt))

    # Build summary text
    parts = []

    # ATT summary
    if att_roll_errors:
        roll_err_mean = sum(att_roll_errors) / len(att_roll_errors)
        parts.append(f"ATT: DesRoll-Roll mean error={roll_err_mean:.3f}")
    if att_pitch_errors:
        pitch_err_mean = sum(att_pitch_errors) / len(att_pitch_errors)
        parts.append(f"ATT: DesPitch-Pitch mean error={pitch_err_mean:.3f}")

    # GPS summary
    if nsats_min is not None:
        parts.append(f"GPS: NSats min={nsats_min}")
    if hdop_max is not None:
        parts.append(f"GPS: HDOP max={hdop_max:.2f}")

    # ERR / EV (limit 10 each)
    if err_messages:
        err_list = err_messages[:10]
        parts.append(f"ERR messages (10 of {len(err_messages)}): " + " | ".join(err_list))
    if ev_messages:
        ev_list = ev_messages[:10]
        parts.append(f"EV messages (10 of {len(ev_messages)}): " + " | ".join(ev_list))

    # CTUN summary
    if ctnu_tho_values:
        tho_avg = sum(ctnu_tho_values) / len(ctnu_tho_values)
        tho_max = max(ctnu_tho_values)
        parts.append(f"CTUN: ThO mean={tho_avg:.2f}, max={tho_max:.2f}")

    # VIBE summary
    if vibe_x_vals:
        parts.append(f"VIBE: VibeX mean={sum(vibe_x_vals)/len(vibe_x_vals):.2f}, max={max(vibe_x_vals):.2f}")
    if vibe_y_vals:
        parts.append(f"VIBE: VibeY mean={sum(vibe_y_vals)/len(vibe_y_vals):.2f}, max={max(vibe_y_vals):.2f}")
    if vibe_z_vals:
        parts.append(f"VIBE: VibeZ mean={sum(vibe_z_vals)/len(vibe_z_vals):.2f}, max={max(vibe_z_vals):.2f}")

    # BAT summary
    if bat_voltages:
        start_volt = bat_voltages[0]
        min_volt = min(bat_voltages)
        parts.append(f"BAT: start voltage={start_volt:.2f}V, min voltage={min_volt:.2f}V")

    if not parts:
        return "No summarizable data found in the log."

    return " | ".join(parts)


def call_openai_api(messages, api_key):
    """
    Call OpenAI API with the conversation messages.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


@st.cache_data(show_spinner=False)
def analyze_log_file(file_bytes: bytes):
    """
    Parse the log file once and cache results.
    This prevents re-parsing on every Streamlit rerun (scroll, map interaction, etc.).
    """
    import tempfile
    from pymavlink import mavutil
    import os
    import pandas as pd
    
    # Write to temp file for pymavlink
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    tfile.write(file_bytes)
    tfile.close()
    file_path = os.path.abspath(tfile.name)
    
    try:
        # Basic file info
        file_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            first_bytes = f.read(16)
        
        # First pass: message types + simple validity flags
        master = mavutil.mavlink_connection(file_path)
        message_types = set()
        message_count = 0
        has_parm = False
        has_msg = False
        err_messages = []
        ev_messages = []
        
        consecutive_none_count = 0
        max_consecutive_none = 500
        i = 0
        
        while True:
            if i >= 1000000000:  # Safety cap
                break
            
            i += 1
            msg = master.recv_match(blocking=False)
            
            if msg is None:
                consecutive_none_count += 1
                if message_count == 0:
                    try:
                        msg = master.recv_match(blocking=True)
                        consecutive_none_count = 0
                    except:
                        break
                else:
                    if consecutive_none_count >= max_consecutive_none:
                        break
                    continue
            
            consecutive_none_count = 0
            
            if msg is not None:
                message_count += 1
                msg_type = msg.get_type()
                message_types.add(msg_type)

                # Simple integrity check: presence of PARM / MSG messages
                if msg_type == "PARM":
                    has_parm = True
                elif msg_type == "MSG":
                    has_msg = True
                elif msg_type == "ERR":
                    if len(err_messages) < 50:
                        err_messages.append(str(msg))
                elif msg_type == "EV":
                    if len(ev_messages) < 50:
                        ev_messages.append(str(msg))
        
        master.close()
        
        # Second pass: GPS track
        gps_points = []
        try:
            master_gps = mavutil.mavlink_connection(file_path)
            consecutive_none_gps = 0
            
            while True:
                msg = master_gps.recv_match(blocking=False)
                if msg is None:
                    consecutive_none_gps += 1
                    if consecutive_none_gps >= 200:
                        break
                    continue
                
                consecutive_none_gps = 0
                mtype = msg.get_type()
                
                if mtype not in ("GPS", "GPS2", "GPS_RAW", "GPS_RAW_INT", "GPS2_RAW"):
                    continue
                
                try:
                    lat = getattr(msg, "Lat", None) or getattr(msg, "lat", None)
                    lon = (
                        getattr(msg, "Lng", None)
                        or getattr(msg, "lon", None)
                        or getattr(msg, "Lon", None)
                    )
                    
                    if lat is None or lon is None:
                        continue
                    
                    if abs(lat) > 90 or abs(lon) > 180:
                        lat = lat / 1e7
                        lon = lon / 1e7
                    
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        gps_points.append((lat, lon))
                except Exception:
                    continue
        except Exception:
            gps_points = []
        
        # Third pass: VIBE data (vibration)
        vibe_data = []
        if 'VIBE' in message_types:
            try:
                master_vibe = mavutil.mavlink_connection(file_path)
                consecutive_none_vibe = 0
                vibe_count = 0
                
                while True:
                    msg = master_vibe.recv_match(type='VIBE', blocking=False)
                    
                    if msg is None:
                        consecutive_none_vibe += 1
                        if consecutive_none_vibe >= 100:
                            break
                        continue
                    
                    consecutive_none_vibe = 0
                    vibe_count += 1
                    
                    try:
                        vibe_x = getattr(msg, 'vibe_x', getattr(msg, 'VibeX', getattr(msg, 'x', None)))
                        vibe_y = getattr(msg, 'vibe_y', getattr(msg, 'VibeY', getattr(msg, 'y', None)))
                        vibe_z = getattr(msg, 'vibe_z', getattr(msg, 'VibeZ', getattr(msg, 'z', None)))
                        timestamp = getattr(msg, 'time_usec', getattr(msg, 'time_boot_ms', vibe_count))
                        
                        if vibe_x is not None or vibe_y is not None or vibe_z is not None:
                            vibe_data.append({
                                'time': timestamp,
                                'x': vibe_x if vibe_x is not None else 0,
                                'y': vibe_y if vibe_y is not None else 0,
                                'z': vibe_z if vibe_z is not None else 0
                            })
                    except Exception:
                        continue
            except Exception:
                vibe_data = []

        # Fourth pass: BAT data (battery voltage/current)
        bat_data = []
        if 'BAT' in message_types:
            try:
                master_bat = mavutil.mavlink_connection(file_path)
                consecutive_none_bat = 0
                bat_count = 0

                while True:
                    msg = master_bat.recv_match(type='BAT', blocking=False)

                    if msg is None:
                        consecutive_none_bat += 1
                        if consecutive_none_bat >= 100:
                            break
                        continue

                    consecutive_none_bat = 0
                    bat_count += 1

                    try:
                        volt = getattr(msg, 'Volt', getattr(msg, 'volt', getattr(msg, 'V', None)))
                        curr = getattr(msg, 'Curr', getattr(msg, 'curr', getattr(msg, 'I', None)))
                        timestamp = getattr(msg, 'time_usec', getattr(msg, 'time_boot_ms', bat_count))

                        if volt is None and curr is None:
                            continue

                        bat_data.append({
                            'time': timestamp,
                            'volt': float(volt) if volt is not None else 0.0,
                            'curr': float(curr) if curr is not None else 0.0,
                        })
                    except Exception:
                        continue
            except Exception:
                bat_data = []
        
        # Process VIBE DataFrame if we have data
        df_vibe = None
        if vibe_data:
            df_vibe = pd.DataFrame(vibe_data)
            if 'time' in df_vibe.columns:
                df_vibe['time_normalized'] = (df_vibe['time'] - df_vibe['time'].min()) / 1000000
            else:
                df_vibe['time_normalized'] = range(len(df_vibe))

        # Process BAT DataFrame if we have data
        df_bat = None
        if bat_data:
            df_bat = pd.DataFrame(bat_data)
            if 'time' in df_bat.columns:
                df_bat['time_normalized'] = (df_bat['time'] - df_bat['time'].min()) / 1000000
            else:
                df_bat['time_normalized'] = range(len(df_bat))
        
        # Generate AI log summary (context injection)
        log_summary = generate_log_summary(file_path)
        
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
            'log_summary': log_summary,
            'err_messages': err_messages,
            'ev_messages': ev_messages,
        }
    
    finally:
        try:
            os.remove(file_path)
        except:
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
    
    if len(message_types) > 0:
        st.success(f"✅ Log parsed successfully (message types found: {len(message_types)})")
        
        visible_limit = 30
        extra_count = max(len(message_types) - visible_limit, 0)
        base_text = ", ".join(message_types[:visible_limit])
        if extra_count > 0:
            st.write(f"**Message types:** {base_text} ... and {extra_count} more")
        else:
            st.write(f"**Message types:** {base_text}")
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
        with col1:
            st.metric("X mean", f"{df_vibe['x'].mean():.2f}")
            st.metric("X max", f"{df_vibe['x'].max():.2f}")
            st.metric("X min", f"{df_vibe['x'].min():.2f}")
        with col2:
            st.metric("Y mean", f"{df_vibe['y'].mean():.2f}")
            st.metric("Y max", f"{df_vibe['y'].max():.2f}")
            st.metric("Y min", f"{df_vibe['y'].min():.2f}")
        with col3:
            st.metric("Z mean", f"{df_vibe['z'].mean():.2f}")
            st.metric("Z max", f"{df_vibe['z'].max():.2f}")
            st.metric("Z min", f"{df_vibe['z'].min():.2f}")
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

    # Check if API key is available via Streamlit secrets
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        st.error("Please contact the administrator (API key configuration error).")
    else:
        # Generate log summary for context injection (if data exists)
        if "analyzed_data" in st.session_state:
            log_summary = st.session_state.analyzed_data.get("log_summary", "")
            system_prompt = f"""You are a world-class Ardupilot Log Analysis Expert with 20 years of experience. Answer in English only. Be professional, concise, and helpful.

Here is the concise summary of this flight:
{log_summary}
Answer the user's questions based on this summary. If the question goes beyond the log, provide general drone operations/maintenance guidance and clearly state any assumptions."""
        else:
            system_prompt = """You are a world-class Ardupilot Log Analysis Expert with 20 years of experience. Answer in English only. Be professional, concise, and helpful.

No log file is available yet. Provide general guidance on log analysis, common anomalies, how to interpret vibration/battery/GPS metrics, and maintenance best practices."""

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
            api_messages = [
                {"role": "system", "content": system_prompt}
            ]
            # Add conversation history (last 10 messages to avoid token limit)
            for msg in st.session_state.messages[-10:]:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            
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
