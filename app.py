import base64
import datetime
import glob
import os
import random
import shutil
import tempfile

import cv2
import pandas as pd
import pydeck as pdk
import pytz
import streamlit as st
from loguru import logger
from ultralytics import YOLO

# Model weight file
MODEL_PATH = "weights/model_v2.pt"
CONF_THRES = 0.75


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
                if "ID:" in line and "Lat:" in line:
                    try:
                        parts = line.strip().split(" | ")
                        eid = int(parts[0].split(": ")[1])
                        time_and_lat = parts[1].split(" Lat: ")
                        ts = time_and_lat[0].split(": ")[1]
                        lat = float(time_and_lat[1])
                        lon = float(parts[2].split(": ")[1])
                        dt_obj = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

                        st.session_state["alerts"].append(
                            {
                                "id": eid,
                                "label": f"#{eid}",
                                "date": dt_obj.strftime("%Y-%m-%d"),
                                "time": dt_obj.strftime("%H:%M:%S"),
                                "latitude": lat,
                                "longitude": lon,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Skipping malformed line: {e}")
                        continue

if "last_processed_file" not in st.session_state:
    st.session_state["last_processed_file"] = None


def get_image_similarity(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        return 0.0
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0, score * 100)


@st.dialog("Confirm Deletion")
def confirm_delete_dialog(alert_id):
    st.warning(f"Are you sure you want to permanently delete Event #{alert_id}?")
    c1, c2 = st.columns(2)
    if c1.button("YES, DELETE", use_container_width=True):
        st.session_state["alerts"] = [
            a for a in st.session_state["alerts"] if a["id"] != alert_id
        ]
        event_folder = os.path.join(IMG_DIR, f"event_{alert_id}")
        if os.path.exists(event_folder):
            shutil.rmtree(event_folder)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                for a in st.session_state["alerts"]:
                    f.write(
                        f"ID: {a['id']} | Time: {a['date']} {a['time']} Lat: {a['latitude']} | Lon: {a['longitude']}\n"
                    )
        st.rerun()
    if c2.button("CANCEL", use_container_width=True):
        st.rerun()


@st.dialog("Reidentification Results")
def reid_dialog(alert_id):
    st.info(f"Scanning Database for Event #{alert_id}...")

    def get_best_img(eid):
        folder = os.path.join(IMG_DIR, f"event_{eid}")
        imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        return imgs[len(imgs) // 2] if imgs else None

    current_img_path = get_best_img(alert_id)
    best_match, highest_score = None, 0

    if current_img_path:
        for alert in st.session_state["alerts"]:
            if alert["id"] == alert_id:
                continue
            compare_path = get_best_img(alert["id"])
            if compare_path:
                score = get_image_similarity(current_img_path, compare_path)
                if score > highest_score:
                    highest_score, best_match = score, alert

    # Not match if lower than 85%
    if best_match and highest_score >= 85.0:
        st.success(
            f"Match Found! {highest_score:.1f}% similar to Event {best_match['label']}"
        )
        col1, col2 = st.columns(2)
        with col1:
            st.image(current_img_path, caption="Current Target")
        with col2:
            st.image(
                get_best_img(best_match["id"]), caption=f"Match ({highest_score:.1f}%)"
            )
    else:
        st.warning("No match found...")


def get_base64_img(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


logo_base64 = get_base64_img("bear_logo_1.png")

st.markdown(
    """
    <style>
        html, body, [data-testid="stWidgetLabel"] p, .stMarkdown p, p, span, li, label {
            font-size: 1rem !important; font-weight: 700 !important; font-family: system-ui, sans-serif !important;
        }
        .art-title-container {
            background: linear-gradient(135deg, #008080 0%, #004d4d 100%);
            padding: 30px; border-radius: 16px; display: flex; align-items: center; justify-content: center; gap: 30px; margin-bottom: 35px;
        }
        .art-title-text { color: white !important; font-size: 3rem !important; font-weight: 900 !important; margin: 0; text-transform: uppercase; letter-spacing: 2px; }
        .logo-upload-label {
            cursor: pointer; transition: transform 0.2s ease; display: inline-block;
        }
        .logo-upload-label:hover { transform: scale(1.1); filter: brightness(1.3); }

        div[data-testid="stTabs"] button {
            font-size: 1.7rem !important; font-weight: 800 !important;
            color: #D7CCC8 !important; background-color: #570f0b  !important;
            padding: 22px 45px !important; border-radius: 14px 14px 0px 0px !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] { background-color: #008080 !important; color: white !important; }

        .bw-table { width: 100%; font-size: 1.5rem !important; border-collapse: collapse; margin-top: 20px; border: 4px solid black; }
        .bw-table th { background-color: #000000; color: #ffffff; padding: 18px; }
        .bw-table td { padding: 18px; border: 1px solid #000000; background-color: #ffffff; color: black; font-weight: 900 !important; }

        div.stButton > button[key^="prev_"], div.stButton > button[key^="next_"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-radius: 50% !important;
            width: 30px !important;
            height: 30px !important;
            min-width: 30px !important;
            padding: 0px !important;
            line-height: 30px !important;
            border: 1px solid #555555 !important;
            font-size: 1rem !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div.stButton > button[key^="del_"] { background-color: #0000FF !important; color: white !important; border: 2px solid white !important; margin-bottom: 10px; }
        div.stButton > button[key^="reid_"] { background-color: #FF8C00 !important; color: black !important; border: 2px solid black !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="art-title-container"><label for="video_upload" class="logo-upload-label"><img src="data:image/png;base64,{logo_base64}" style="max-height:150px;"></label><h1 class="art-title-text">Intelligent Bear Surveillance Platform</h1></div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload Video:",
    type=["mp4", "mov", "avi"],
    key="video_upload",
    label_visibility="collapsed",
)

if uploaded_file and st.session_state["last_processed_file"] != uploaded_file.name:
    with st.spinner("Processing video for bear detection. This may take a moment..."):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        progress_bar, total_frames, frame_count = (
            st.progress(0),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            0,
        )
        eid = len(st.session_state["alerts"]) + 1
        event_folder = os.path.join(IMG_DIR, f"event_{eid}")
        os.makedirs(event_folder, exist_ok=True)
        detections_found = False

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
                        bear_detected = detections_found = True
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
            if bear_detected:
                # Resize image to half-size
                small_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                cv2.imwrite(
                    os.path.join(event_folder, f"frame_{frame_count:04d}.jpg"),
                    small_frame,
                )

            # Update the progress bar and image only every 5 frames
            if frame_count % 5 == 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                frame_placeholder.image(frame, channels="BGR")

        if detections_found:
            now = datetime.datetime.now(datetime.timezone.utc).astimezone(
                pytz.timezone("Asia/Tokyo")
            )
            new_alert = {
                "id": eid,
                "label": f"#{eid}",
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "latitude": 39.72 + (eid * 0.006),
                "longitude": 140.10 + (random.uniform(0.004, 0.012)),
            }
            with open(LOG_FILE, "a") as f:
                f.write(
                    f"ID: {eid} | Time: {now.strftime('%Y-%m-%d %H:%M:%S')} Lat: {new_alert['latitude']} | Lon: {new_alert['longitude']}\n"
                )
            st.session_state["alerts"].append(new_alert)
        cap.release()
        st.session_state["last_processed_file"] = uploaded_file.name
        st.rerun()

tabs = st.tabs(["🐾 SYSTEM ALERTS", "🌐 DISTRIBUTION MAP", "📊 HISTORY"])

with tabs[0]:
    st.markdown("### Preview Alert Snapshots🐾 ")
    if not st.session_state["alerts"]:
        st.write("Upload video...")
    else:
        for alert in reversed(st.session_state["alerts"]):
            with st.container(border=True):
                main_cols = st.columns([1.8, 1.2, 0.8])
                event_folder = os.path.join(IMG_DIR, f"event_{alert['id']}")
                images = sorted(glob.glob(os.path.join(event_folder, "*.jpg")))

                if images:
                    idx_key = f"idx_{alert['id']}"
                    if idx_key not in st.session_state:
                        st.session_state[idx_key] = 0

                    with main_cols[0]:
                        st.image(
                            images[st.session_state[idx_key]], use_container_width=False
                        )

                        c_prev, c_count, c_next, _ = st.columns([1, 3, 1, 4])

                        if c_prev.button("◁", key=f"prev_{alert['id']}"):
                            st.session_state[idx_key] = (
                                st.session_state[idx_key] - 1
                            ) % len(images)
                            st.rerun()

                        c_count.markdown(
                            f"""
                            <p style='text-align:center; 
                                    color:blue; 
                                    font-size:1.8rem; 
                                    font-weight:bold; 
                                    margin:0;'>
                                {st.session_state[idx_key] + 1} / {len(images)}
                            </p>
                            """,
                            unsafe_allow_html=True,
                        )

                        if c_next.button("▷", key=f"next_{alert['id']}"):
                            st.session_state[idx_key] = (
                                st.session_state[idx_key] + 1
                            ) % len(images)
                            st.rerun()

                main_cols[1].markdown(
                    f"### EVENT {alert['label']}\n**DATE:** {alert['date']}\n| **TIME:** {alert['time']}"
                )

                with main_cols[2]:
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
        map_df = pd.DataFrame(st.session_state["alerts"])

        def get_mid_img(eid):
            imgs = sorted(glob.glob(os.path.join(IMG_DIR, f"event_{eid}", "*.jpg")))
            return imgs[len(imgs) // 2] if imgs else ""

        map_df["img_b64"] = map_df["id"].apply(
            lambda x: f"data:image/jpeg;base64,{get_base64_img(get_mid_img(x))}"
        )
        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
                initial_view_state=pdk.ViewState(
                    latitude=map_df["latitude"].mean(),
                    longitude=map_df["longitude"].mean(),
                    zoom=12,
                ),
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
                    "html": "<b>EVENT {label}</b><br/>{date} {time}<br/><img src='{img_b64}' style='width:200px;'/>"
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
