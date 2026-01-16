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
st.title("🚁 AI‑based Ardupilot Log Analyzer")
st.write("Upload an Ardupilot log file (.bin) to inspect basic info and vibration data.")

# Load OpenAI API key from secrets
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    model_choice = "gpt-4o-mini"  # Cost-effective model
except (KeyError, FileNotFoundError):
    api_key = None
    model_choice = None


def generate_log_summary(data):
    """
    Generate a summary of analyzed log data for context injection into AI prompts.
    """
    summary_parts = []
    
    summary_parts.append(f"Log file analyzed: {data['message_count']:,} messages parsed, {len(data['message_types'])} message types found.")
    
    if data.get('has_parm') or data.get('has_msg'):
        summary_parts.append("Log contains PARM/MSG records (typical Ardupilot DataFlash format).")
    
    # Vibration data summary
    if data.get('df_vibe') is not None and len(data['df_vibe']) > 0:
        df_vibe = data['df_vibe']
        vibe_x_mean = df_vibe['x'].mean()
        vibe_y_mean = df_vibe['y'].mean()
        vibe_z_mean = df_vibe['z'].mean()
        vibe_x_max = df_vibe['x'].max()
        vibe_y_max = df_vibe['y'].max()
        vibe_z_max = df_vibe['z'].max()
        summary_parts.append(
            f"Vibration data: X-axis avg={vibe_x_mean:.2f} (max={vibe_x_max:.2f}), "
            f"Y-axis avg={vibe_y_mean:.2f} (max={vibe_y_max:.2f}), "
            f"Z-axis avg={vibe_z_mean:.2f} (max={vibe_z_max:.2f})."
        )
    
    # Battery data summary
    if data.get('df_bat') is not None and len(data['df_bat']) > 0:
        df_bat = data['df_bat']
        min_volt = df_bat['volt'].min()
        max_volt = df_bat['volt'].max()
        avg_volt = df_bat['volt'].mean()
        max_curr = df_bat['curr'].max()
        avg_curr = df_bat['curr'].mean()
        summary_parts.append(
            f"Battery data: Voltage range {min_volt:.2f}V - {max_volt:.2f}V (avg={avg_volt:.2f}V), "
            f"Current max={max_curr:.2f}A (avg={avg_curr:.2f}A)."
        )
    
    # GPS track summary
    if data.get('gps_points') and len(data['gps_points']) >= 2:
        gps_count = len(data['gps_points'])
        summary_parts.append(f"GPS track: {gps_count} waypoints recorded.")
    
    return " ".join(summary_parts)


def call_openai_api(messages, api_key, model="gpt-4o"):
    """
    Call OpenAI API with the conversation messages.
    """
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
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
    
    # GPS track: draw flight path on a map using folium
    st.write("---")
    st.subheader("🗺️ Flight path (GPS)")
    
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

# AI Chatbot section (always visible, even without file upload)
st.write("---")
st.subheader("🤖 AI Drone Consultation Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if API key is available from secrets
if not api_key:
    st.error("⚠️ API 키 설정 오류: 관리자에게 문의하세요.")
else:
    # Generate log summary for context injection (if data exists)
    if "analyzed_data" in st.session_state:
        log_summary = generate_log_summary(st.session_state.analyzed_data)
        system_prompt = f"""You are an expert drone flight log analyst and consultant. Your role is to help users understand their Ardupilot log data and provide actionable advice.

Current log analysis results:
{log_summary}

Provide clear, technical, and helpful responses about:
- Flight log interpretation
- Potential issues or anomalies detected
- Recommendations for drone maintenance or flight improvements
- Explanation of technical metrics (vibration, battery, GPS, etc.)

Answer in a professional but friendly manner. If the user asks about something not in the log data, acknowledge it and provide general guidance."""
    else:
        system_prompt = """You are an expert drone flight log analyst and consultant. Your role is to help users understand Ardupilot log data and provide actionable advice.

The user has not uploaded a log file yet, or the file is being processed. You can still provide general guidance about:
- Ardupilot log analysis
- Common issues in drone flight logs
- How to interpret various metrics (vibration, battery, GPS, etc.)
- Best practices for drone maintenance

Answer in a professional but friendly manner."""

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
                response = call_openai_api(api_messages, api_key, model_choice)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()