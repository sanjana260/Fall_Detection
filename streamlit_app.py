"""
Vigilens UI — Streamlit wrapper around fall_detection_live_copy.py.

Runs the EXACT original detection function untouched. The UI captures
frames via a monkey-patched cv2.imshow and watches the output directory
for fall events. Video updates every ~2 seconds (with streamlit-autorefresh).

Run:  streamlit run streamlit_app.py   (from the Fall_Detection directory)
Open: http://localhost:8501
"""

import sys
import os
import json
import time
import threading
import glob
from pathlib import Path
from datetime import datetime
import cv2
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

# ═══════════════════════════════════════════════════════════════════
# FRAME CAPTURE — monkey-patch cv2.imshow before importing detection
# ═══════════════════════════════════════════════════════════════════

ui_frame_jpeg = None
ui_frame_lock = threading.Lock()

_original_imshow = cv2.imshow


def _patched_imshow(winname, frame):
    global ui_frame_jpeg
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with ui_frame_lock:
        ui_frame_jpeg = buf.tobytes()


cv2.imshow = _patched_imshow

_original_waitKey = cv2.waitKey


def _patched_waitKey(delay=0):
    time.sleep(max(delay, 1) / 1000.0)
    return -1


cv2.waitKey = _patched_waitKey

# ═══════════════════════════════════════════════════════════════════
# IMPORT DETECTION MODULE (without triggering its module-level call)
# ═══════════════════════════════════════════════════════════════════

_source_path = SCRIPT_DIR / "fall_detection_live_copy.py"
_source = _source_path.read_text()

_filtered_lines = []
_in_main_guard = False
for _line in _source.split("\n"):
    stripped = _line.strip()
    if stripped == 'if __name__ == "__main__":':
        _in_main_guard = True
        _filtered_lines.append(_line)
        continue
    if stripped.startswith("output_dir, people = run_fall_detection_live"):
        if _in_main_guard:
            _filtered_lines.append(_line.replace(stripped, "pass  # skipped by UI wrapper"))
        else:
            _filtered_lines.append("# " + _line + "  # skipped by UI wrapper")
        _in_main_guard = False
        continue
    _in_main_guard = False
    _filtered_lines.append(_line)

_namespace = {"__name__": "fall_detection_live_copy", "__file__": str(_source_path)}
exec("\n".join(_filtered_lines), _namespace)

run_fall_detection_live = _namespace["run_fall_detection_live"]

# ═══════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════

fall_events = []
fall_events_lock = threading.Lock()
_threads_started = False
_threads_lock = threading.Lock()

OUTPUT_STEM = "vigilens_live"
FALL_DIR = os.path.join("Outputs", OUTPUT_STEM, "falls")
RECOVERIES_FILE = "recoveries.txt"

# ═══════════════════════════════════════════════════════════════════
# FILE WATCHERS
# ═══════════════════════════════════════════════════════════════════


def watch_falls():
    """Watch FALL_DIR for new .json files and append to fall_events."""
    seen_files = set()
    fall_count = 0

    while True:
        time.sleep(0.5)
        if not os.path.isdir(FALL_DIR):
            continue

        json_files = sorted(glob.glob(os.path.join(FALL_DIR, "*.json")))
        for jf in json_files:
            if jf in seen_files:
                continue

            try:
                with open(jf, "r") as f:
                    raw = json.load(f)
                if isinstance(raw, str):
                    fall_data = json.loads(raw)
                else:
                    fall_data = raw
            except Exception:
                continue

            seen_files.add(jf)
            fall_count += 1

            screenshot_path = None
            video_path = fall_data.get("fall_video")

            if video_path and os.path.isfile(video_path):
                cap_temp = cv2.VideoCapture(video_path)
                if cap_temp.isOpened():
                    ret, thumb = cap_temp.read()
                    if ret:
                        screenshot_path = jf.replace(".json", "_thumb.jpg")
                        cv2.imwrite(screenshot_path, thumb)
                    cap_temp.release()

            frames_data = fall_data.get("frames", [])
            serializable_frames = []
            for fr in frames_data:
                clean = {k: v for k, v in fr.items() if k != "frame"}
                serializable_frames.append(clean)

            ui_evt = {
                "id": fall_count,
                "person_id": fall_data.get("person_id"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "active",
                "fall_data": {
                    "fall_id": fall_data.get("fall_id"),
                    "person_id": fall_data.get("person_id"),
                    "fall_frame": fall_data.get("fall_frame"),
                    "head_speed": fall_data.get("head_speed"),
                    "head_accel": fall_data.get("head_accel"),
                    "shoulder_speed": fall_data.get("shoulder_speed"),
                    "fall_start": fall_data.get("fall_start"),
                    "fall_end": fall_data.get("fall_end"),
                    "fall_video": fall_data.get("fall_video"),
                    "fall_json": fall_data.get("fall_json"),
                    "frames": serializable_frames,
                },
                "detection": {},
                "fall_object_interaction": fall_data.get("fall_object_interaction"),
                "_clip_path": video_path,
                "_screenshot_path": screenshot_path,
                "_json_path": jf,
            }

            with fall_events_lock:
                fall_events.append(ui_evt)


def watch_recoveries():
    """Watch recoveries.txt for new recovery lines."""
    seen_lines = 0

    if os.path.isfile(RECOVERIES_FILE):
        with open(RECOVERIES_FILE, "r") as f:
            seen_lines = len(f.readlines())

    while True:
        time.sleep(0.5)
        if not os.path.isfile(RECOVERIES_FILE):
            continue

        with open(RECOVERIES_FILE, "r") as f:
            lines = f.readlines()

        if len(lines) <= seen_lines:
            continue

        new_lines = lines[seen_lines:]
        seen_lines = len(lines)

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1].lower() == "recovered":
                person_id = parts[0]

                with fall_events_lock:
                    for evt in reversed(fall_events):
                        if str(evt.get("person_id")) == str(person_id) and evt["status"] == "active":
                            evt["status"] = "recovered"
                            evt["recovered_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            break


def watch_enrichment():
    """Re-read fall JSONs periodically to pick up fall_object_interaction data."""
    enriched = set()

    while True:
        time.sleep(2)
        with fall_events_lock:
            events_copy = list(fall_events)

        for evt in events_copy:
            eid = evt["id"]
            if eid in enriched:
                continue
            jpath = evt.get("_json_path")
            if not jpath or not os.path.isfile(jpath):
                continue

            try:
                with open(jpath, "r") as f:
                    raw = json.load(f)
                if isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = raw
            except Exception:
                continue

            foi = data.get("fall_object_interaction")
            if foi and evt.get("fall_object_interaction") != foi:
                enriched.add(eid)
                with fall_events_lock:
                    evt["fall_object_interaction"] = foi


# ═══════════════════════════════════════════════════════════════════
# DETECTION THREAD
# ═══════════════════════════════════════════════════════════════════


def detection_thread():
    if os.path.isfile(RECOVERIES_FILE):
        os.remove(RECOVERIES_FILE)
    run_fall_detection_live(0, output_filename=OUTPUT_STEM + ".mp4")


# ═══════════════════════════════════════════════════════════════════
# START BACKGROUND THREADS (once)
# ═══════════════════════════════════════════════════════════════════


def maybe_start_threads():
    global _threads_started
    with _threads_lock:
        if _threads_started:
            return
        _threads_started = True
        threading.Thread(target=detection_thread, daemon=True).start()
        threading.Thread(target=watch_falls, daemon=True).start()
        threading.Thread(target=watch_recoveries, daemon=True).start()
        threading.Thread(target=watch_enrichment, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════


def fmt_num(v):
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    n = None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{n:.2f}" if not (n != n) else str(v)


def format_time(ts):
    if not ts:
        return ""
    parts = str(ts).split(" ")
    return parts[1] if len(parts) > 1 else ts


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Vigilens", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS — #F3F8FF light theme
st.markdown(
    """
<style>
    .stApp { background: #F3F8FF; }
    div[data-testid="stHorizontalBlock"] { gap: 1rem; }
    .vigilens-header { display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; }
    .vigilens-logo { font-size: 1.4rem; font-weight: 700; color: #1a1f36; }
    .vigilens-badge { font-size: 0.75rem; padding: 4px 12px; border-radius: 20px; background: #EAF0FA; color: #5e6687; }
    .fall-card { background: rgba(255,255,255,0.9); border-radius: 12px; padding: 14px; margin-bottom: 10px; border: 1px solid rgba(60,80,130,0.12); }
    .fall-card.active { border-color: rgba(220,38,38,0.3); }
    .fall-card.recovered { border-color: rgba(217,119,6,0.25); }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }
    .detail-item { padding: 8px 10px; background: #EAF0FA; border-radius: 8px; font-size: 0.85rem; }
    .detail-label { font-size: 0.7rem; color: #8e94b0; text-transform: uppercase; }
    .risk-high { background: rgba(220,38,38,0.1); color: #dc2626; padding: 2px 6px; border-radius: 4px; }
    .risk-medium { background: rgba(217,119,6,0.1); color: #d97706; padding: 2px 6px; border-radius: 4px; }
    .risk-low { background: rgba(5,150,105,0.1); color: #059669; padding: 2px 6px; border-radius: 4px; }
</style>
""",
    unsafe_allow_html=True,
)

maybe_start_threads()

# Auto-refresh every 2 seconds (if streamlit-autorefresh is installed)
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="vigilens_refresh")

# Session state for toast notifications
if "last_fall_ids" not in st.session_state:
    st.session_state.last_fall_ids = set()
if "last_recovered_ids" not in st.session_state:
    st.session_state.last_recovered_ids = set()

with fall_events_lock:
    events_copy = list(fall_events)

# Toast for new falls
current_ids = {e["id"] for e in events_copy}
new_ids = current_ids - st.session_state.last_fall_ids
for eid in new_ids:
    evt = next((e for e in events_copy if e["id"] == eid), None)
    if evt:
        st.toast(f"⚠️ Fall detected — Person {evt.get('person_id', '?')}", icon="⚠️")
st.session_state.last_fall_ids = current_ids

# Toast for recoveries
recovered_now = {e["id"] for e in events_copy if e.get("status") == "recovered"}
new_recovered = recovered_now - st.session_state.last_recovered_ids
for eid in new_recovered:
    evt = next((e for e in events_copy if e["id"] == eid), None)
    if evt:
        st.toast(f"✓ Person {evt.get('person_id', '?')} recovered — false alarm", icon="✓")
st.session_state.last_recovered_ids = recovered_now

# Header (events_copy already populated above)
st.markdown(
    '<div class="vigilens-header">'
    '<span class="vigilens-logo">Vigilens</span>'
    '<span class="vigilens-badge">LIVE</span>'
    "</div>",
    unsafe_allow_html=True,
)

# Alert banner for active falls (events_copy from above)
active_falls = [e for e in events_copy if e.get("status") == "active"]
if active_falls:
    person_ids = ", ".join(str(e.get("person_id", "?")) for e in active_falls)
    st.error(f"⚠️ FALL DETECTED — Person(s) {person_ids}")

col_feed, col_events = st.columns([3, 2])

with col_feed:
    st.subheader("Live Feed")
    with ui_frame_lock:
        frame_bytes = ui_frame_jpeg
    if frame_bytes is not None:
        st.image(frame_bytes, use_container_width=True)
    else:
        st.info("Waiting for video feed... (webcam starting)")

with col_events:
    st.subheader("Fall Events")
    st.caption(f"{len(events_copy)} event(s)")

    if not events_copy:
        st.info("No falls detected yet. Monitoring webcam feed...")
    else:
        for evt in reversed(events_copy):
            fd = evt.get("fall_data") or {}
            foi = evt.get("fall_object_interaction")
            status = evt.get("status", "active")
            card_class = "active" if status == "active" else "recovered"
            time_str = format_time(evt.get("timestamp", ""))
            person_label = f"Person {evt.get('person_id', '?')}" if evt.get("person_id") is not None else ""

            with st.expander(
                f"#{evt['id']} Fall — {time_str} {person_label} [{status.title()}]",
                expanded=(status == "active"),
            ):
                # Status
                st.write(f"**Status:** {status.title()}")
                if evt.get("recovered_at"):
                    st.write(f"**Recovered at:** {format_time(evt['recovered_at'])}")

                # Screenshot
                thumb_path = evt.get("_screenshot_path")
                if thumb_path and os.path.isfile(thumb_path):
                    st.image(thumb_path, use_container_width=True)

                # Detection & Kinematics
                det = evt.get("detection") or {}
                st.markdown("**Detection & Kinematics**")
                k1, k2 = st.columns(2)
                with k1:
                    if det.get("angle") is not None:
                        st.metric("Torso Angle", f"{det['angle']}°")
                    if fd.get("head_speed") is not None:
                        st.metric("Head Speed", f"{fmt_num(fd['head_speed'])} px/s")
                    if fd.get("head_accel") is not None:
                        st.metric("Head Accel", f"{fmt_num(fd['head_accel'])} px/s²")
                with k2:
                    if det.get("box_ratio") is not None:
                        st.metric("Box Ratio", det["box_ratio"])
                    if fd.get("shoulder_speed") is not None:
                        st.metric("Shoulder Speed", f"{fmt_num(fd['shoulder_speed'])} px/s")

                # Fall Timing
                st.markdown("**Fall Timing**")
                t1, t2 = st.columns(2)
                with t1:
                    if fd.get("fall_frame") is not None:
                        st.metric("Fall Frame", fd["fall_frame"])
                    if fd.get("fall_start"):
                        fs = fd["fall_start"] if isinstance(fd["fall_start"], list) else [fd["fall_start"]]
                        st.metric("Fall Start", f"Frame {fs[0]}")
                with t2:
                    if fd.get("person_id") is not None:
                        st.metric("Person ID", fd["person_id"])
                    if fd.get("fall_end"):
                        fe = fd["fall_end"] if isinstance(fd["fall_end"], list) else [fd["fall_end"]]
                        st.metric("Fall End", f"Frame {fe[0]}")

                # Fall Context (object interaction)
                if foi:
                    st.markdown("**Fall Context**")
                    if foi.get("pre_fall_posture"):
                        st.write(f"Pre-fall posture: {foi['pre_fall_posture']}")
                    if foi.get("fall_time_sec") is not None:
                        st.write(f"Fall time: {foi['fall_time_sec']}s")
                    if foi.get("first_floor_contact"):
                        ffc = foi["first_floor_contact"]
                        if isinstance(ffc, dict):
                            st.write(f"Floor contact: {ffc.get('body_part', 'unknown')}")
                        else:
                            st.write(f"Floor contact: {ffc}")

                    if foi.get("object_interactions"):
                        st.markdown("**Object Interactions**")
                        for inter in foi["object_interactions"]:
                            risk = (inter.get("injury_risk") or "N/A").lower()
                            risk_cls = f"risk-{risk}" if risk in ("high", "medium", "low") else ""
                            parts = []
                            if inter.get("body_parts"):
                                parts.append(", ".join(inter["body_parts"]))
                            if inter.get("proximity_px") is not None:
                                parts.append(f"{inter['proximity_px']}px")
                            extra = " | ".join(parts) if parts else ""
                            st.write(f"- **{inter.get('object', '?')}** [{inter.get('injury_risk', 'N/A')}] {extra}")

                # Frames table
                frames = fd.get("frames") or []
                if frames:
                    st.markdown(f"**Frames Data** ({len(frames)} frames)")
                    rows = []
                    for fr in frames:
                        rows.append({
                            "Frame": fr.get("frame_id", "-"),
                            "Time": fmt_num(fr.get("timestamp")) if fr.get("timestamp") is not None else "-",
                            "Angle": fmt_num(fr.get("angle")) if fr.get("angle") is not None else "-",
                            "ΔAngle": fmt_num(fr.get("angle_change")) if fr.get("angle_change") is not None else "-",
                            "Head Y": fmt_num(fr.get("head_y")) if fr.get("head_y") is not None else "-",
                            "Shoulder Y": fmt_num(fr.get("shoulder_y")) if fr.get("shoulder_y") is not None else "-",
                            "ΔHead": fmt_num(fr.get("head_change")) if fr.get("head_change") is not None else "-",
                            "Horiz": fr.get("horizontal", "-"),
                            "Vert": fr.get("vertical", "-"),
                            "Ongoing": fr.get("fall_ongoing", "-"),
                        })
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)

                # Video clip
                clip_path = evt.get("_clip_path")
                if clip_path and os.path.isfile(clip_path):
                    st.video(clip_path)
                else:
                    st.caption("Recording clip...")
