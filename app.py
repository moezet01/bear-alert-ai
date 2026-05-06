import base64
import datetime
import os
import tempfile

import cv2
import pandas as pd
import pydeck as pdk
import streamlit as st
from ultralytics import YOLO

# Model weight file
MODEL_PATH = "weights/best.pt"
CONF_THRES = 0.4


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()
st.set_page_config(page_title="Bear Detection System", layout="wide")

# Database and directory setup
DB_DIR = "database"
IMG_DIR = os.path.join(DB_DIR, "snapshots")
LOG_FILE = os.path.join(DB_DIR, "history.txt")

for path in [DB_DIR, IMG_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# Session state
if "alerts" not in st.session_state:
    st.session_state["alerts"] = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            for line in f:
                if "ID:" in line:
                    try:
                        parts = line.strip().split(" | ")
                        eid = int(parts[0].split(": ")[1])
                        ts = parts[1].split(": ")[1]
                        dt_obj = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                        st.session_state["alerts"].append(
                            {
                                "id": eid,
                                "label": f"#{eid}",
                                "date": dt_obj.strftime("%Y-%m-%d"),
                                "time": dt_obj.strftime("%H:%M:%S"),
                                "latitude": 35.68 + (eid * 0.005),
                                "longitude": 139.76 + (eid * 0.005),
                            }
                        )
                    except:
                        continue

if "last_processed_file" not in st.session_state:
    st.session_state["last_processed_file"] = None


# Calculate simple bear similarity
def get_image_similarity(img_path1, img_path2):
    """Calculates real visual similarity using HSV Histogram Correlation."""
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        return 0.0

    # Convert to HSV (better for color matching in nature)
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Calculate histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    # Compare using Correlation (range -1 to 1, we return 0-100%)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0, score * 100)


# Dialog box
@st.dialog("Confirm Deletion")
def confirm_delete_dialog(alert_id):
    st.warning(f"Are you sure you want to permanently delete Event #{alert_id}?")
    c1, c2 = st.columns(2)
    if c1.button("YES, DELETE", use_container_width=True):
        st.session_state["alerts"] = [
            a for a in st.session_state["alerts"] if a["id"] != alert_id
        ]
        img_path = os.path.join(IMG_DIR, f"event_{alert_id}.jpg")
        if os.path.exists(img_path):
            os.remove(img_path)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                for a in st.session_state["alerts"]:
                    f.write(f"ID: {a['id']} | Time: {a['date']} {a['time']}\n")
        st.rerun()
    if c2.button("CANCEL", use_container_width=True):
        st.rerun()


@st.dialog("Reidentification Results")
def reid_dialog(alert_id):
    st.info(f"Scanning Database for Event #{alert_id}...")

    current_img_path = os.path.join(IMG_DIR, f"event_{alert_id}.jpg")
    best_match = None
    highest_score = 0

    # Iterate through database to find the real best match
    for alert in st.session_state["alerts"]:
        if alert["id"] == alert_id:
            continue

        compare_path = os.path.join(IMG_DIR, f"event_{alert['id']}.jpg")
        if os.path.exists(compare_path):
            score = get_image_similarity(current_img_path, compare_path)
            if score > highest_score:
                highest_score = score
                best_match = alert

    # Not match if lower than 85%
    if best_match and highest_score >= 85.0:
        st.success(
            f"Match Found! This bear is {highest_score:.1f}% similar to Event {best_match['label']}"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.image(current_img_path, caption="Current Target")
        with col2:
            st.image(
                os.path.join(IMG_DIR, f"event_{best_match['id']}.jpg"),
                caption=f" Match ({highest_score:.1f}%)",
            )
    else:
        st.warning(
            f"No match found. (Highest similarity was {highest_score:.1f}%, which is below the 85% requirement)"
        )


def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


logo_base64 = get_base64_img("bear_logo_1.png")

st.markdown(
    """
    <style>
        html, body, [data-testid="stWidgetLabel"] p, .stMarkdown p, p, span, li, label {
            font-size: 1.3rem !important; font-weight: 700 !important; font-family: system-ui, sans-serif !important;
        }
        .art-title-container {
            background: linear-gradient(135deg, #008080 0%, #004d4d 100%);
            padding: 30px; border-radius: 16px; display: flex; align-items: center; justify-content: center; gap: 30px; margin-bottom: 35px;
        }
        .art-title-text { color: white !important; font-size: 3rem !important; font-weight: 900 !important; margin: 0; text-transform: uppercase; letter-spacing: 2px; }

        .logo-upload-label {
            cursor: pointer; transition: transform 0.2s ease; display: inline-block;
        }
        .logo-upload-label:hover { transform: scale(1.1); filter: brightness(1.2); }

        div[data-testid="stTabs"] button {
            font-size: 1.7rem !important; font-weight: 800 !important;
            color: #D7CCC8 !important; background-color: #3E2723 !important;
            padding: 22px 45px !important; border-radius: 14px 14px 0px 0px !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] { background-color: #008080 !important; color: white !important; }

        .bw-table { width: 100%; font-size: 1.5rem !important; border-collapse: collapse; margin-top: 20px; border: 4px solid black; }
        .bw-table th { background-color: #000000; color: #ffffff; padding: 18px; }
        .bw-table td { padding: 18px; border: 1px solid #000000; background-color: #ffffff; color: black; font-weight: 900 !important; }

        div.stButton > button[key^="del_"] {
            background-color: #0000FF !important; color: white !important; border: 2px solid white !important; margin-bottom: 10px;
        }
        div.stButton > button[key^="reid_"] {
            background-color: #FF8C00 !important; color: black !important; border: 2px solid black !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="art-title-container">
        <label for="video_upload" class="logo-upload-label">
            <img src="data:image/png;base64,{logo_base64}" style="max-height:150px;">
        </label>
        <h1 class="art-title-text">Intelligent Bear Surveillance Platform</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Video:",
    type=["mp4", "mov", "avi"],
    key="video_upload",
    label_visibility="collapsed",
)


if uploaded_file and st.session_state["last_processed_file"] != uploaded_file.name:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    alert_saved = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        results = model(frame, conf=CONF_THRES, verbose=False)
        bear_detected = False
        for result in results:
            for box in result.boxes:
                if model.names[int(box.cls[0])] == "bear":
                    bear_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
                    cv2.putText(
                        frame,
                        "BEAR ALERT",
                        (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.8,
                        (0, 0, 255),
                        5,
                    )

        if bear_detected and not alert_saved:
            now = datetime.datetime.now()
            eid = len(st.session_state["alerts"]) + 1

            # Create a zig-zag GPS points
            direction = 1 if eid % 2 == 0 else -1
            lat_offset = eid * 0.005
            lon_offset = direction * 0.008  # Swings left and right
            new_alert = {
                "id": eid,
                "label": f"#{eid}",
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "latitude": 35.68 + lat_offset,
                "longitude": 139.76 + lon_offset,
            }
            cv2.imwrite(os.path.join(IMG_DIR, f"event_{eid}.jpg"), frame)
            with open(LOG_FILE, "a") as f:
                f.write(f"ID: {eid} | Time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            st.session_state["alerts"].append(new_alert)
            alert_saved = True

        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
        frame_placeholder.image(frame, channels="BGR")
    cap.release()
    st.session_state["last_processed_file"] = uploaded_file.name
    st.rerun()


# UI Tabs
tabs = st.tabs(["🐾 SYSTEM ALERTS", "🌐 DISTRIBUTION MAP", "📊 HISTORY"])

with tabs[0]:
    st.markdown("### Alert Snapshots🐾 ")
    if not st.session_state["alerts"]:
        st.write("Upload the video to process...")
    else:
        for alert in reversed(st.session_state["alerts"]):
            with st.container(border=True):
                c1, c2, c3 = st.columns([1.5, 2, 1.0])
                img_p = os.path.join(IMG_DIR, f"event_{alert['id']}.jpg")
                if os.path.exists(img_p):
                    c1.image(img_p)
                c2.markdown(f"### EVENT {alert['label']}")
                c2.write(f"**DATE:** {alert['date']} | **TIME:** {alert['time']}")
                with c3:
                    if st.button(
                        "DELETE", key=f"del_{alert['id']}", use_container_width=True
                    ):
                        confirm_delete_dialog(alert["id"])
                    if st.button(
                        "REIDENTIFY",
                        key=f"reid_{alert['id']}",
                        use_container_width=True,
                    ):
                        reid_dialog(alert["id"])

with tabs[1]:
    if st.session_state["alerts"]:
        # Prepare Dataframe with Base64 images for Hover preview
        map_df = pd.DataFrame(st.session_state["alerts"])
        map_df["img_b64"] = map_df["id"].apply(
            lambda x: f"data:image/jpeg;base64,{get_base64_img(os.path.join(IMG_DIR, f'event_{x}.jpg'))}"
        )

        view_state = pdk.ViewState(
            latitude=map_df["latitude"].mean(),
            longitude=map_df["longitude"].mean(),
            zoom=12,
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                initial_view_state=view_state,
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        map_df,
                        get_position="[longitude, latitude]",
                        get_color="[255, 0, 0, 200]",
                        get_radius=300,
                        pickable=True,
                    ),
                    pdk.Layer(
                        "TextLayer",
                        map_df,
                        get_position="[longitude, latitude]",
                        get_text="label",
                        get_size=24,
                        get_color=[0, 0, 0],
                        get_alignment_baseline="'bottom'",
                    ),
                ],
                tooltip={
                    "html": """
                <div style="background-color: white; color: black; padding: 10px; border: 2px solid black; border-radius: 5px; text-align: center;">
                    <b>EVENT {label}</b><br/>
                    {date} {time}<br/>
                    <img src="{img_b64}" style="width:210px; margin-top:5px; border-radius: 3px; border: 1px solid #ccc;"/>
                </div>
                """,
                    "style": {
                        "backgroundColor": "transparent",
                        "color": "white",
                        "zIndex": "10000",
                    },
                },
            )
        )

with tabs[2]:
    if st.session_state["alerts"]:
        sort_choice = st.radio(" ", ["NEWEST", "OLDEST"], horizontal=True)
        df_hist = pd.DataFrame(st.session_state["alerts"])
        df_hist = df_hist.sort_values(by="id", ascending=(sort_choice == "OLDEST"))
        html_table = "<table class='bw-table'><thead><tr><th>ID</th><th>DATE</th><th>TIME</th><th>LATITUDE</th><th>LONGITUDE</th></tr></thead><tbody>"
        for _, row in df_hist.iterrows():
            html_table += f"<tr><td>{row['label']}</td><td>{row['date']}</td><td>{row['time']}</td><td>{row['latitude']:.4f}</td><td>{row['longitude']:.4f}</td></tr>"
        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)

        # Download detected alerts as CSV file
        csv = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="bear_detection_history.csv",
            mime="text/csv",
        )
