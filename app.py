"""
Vigilens UI — Flask wrapper around fall_detection_live_copy.py.

Runs the EXACT original detection function untouched.  The UI captures
frames via a monkey-patched cv2.imshow and watches the output directory
for new fall JSON files and recoveries.txt for recovery events.

Run:  python app.py      (from the Fall_Detection directory)
Open: http://localhost:8000
"""

import sys
import os
import json
import time
import threading
import glob
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime

import cv2
from flask import Flask, Response, render_template, jsonify, send_file

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

os.chdir("/Users/nikitha/Desktop/Fall_Detection")
sys.path.insert(0, "/Users/nikitha/Desktop/Fall_Detection")

_source_path = Path("/Users/nikitha/Desktop/Fall_Detection/fall_detection_live_copy.py")
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
# SSE EVENT BUS
# ═══════════════════════════════════════════════════════════════════

sse_queues = []
sse_lock = threading.Lock()
fall_events = []
fall_events_lock = threading.Lock()

OUTPUT_STEM = "vigilens_live"
FALL_DIR = os.path.join("Outputs", OUTPUT_STEM, "falls")
RECOVERIES_FILE = "recoveries.txt"


def push_sse(data):
    payload = json.dumps(data)
    with sse_lock:
        dead = []
        for q in sse_queues:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            sse_queues.remove(q)


# ═══════════════════════════════════════════════════════════════════
# FILE WATCHERS — detect new falls and recoveries from disk output
# ═══════════════════════════════════════════════════════════════════

def watch_falls():
    """Watch FALL_DIR for new .json files and push SSE events."""
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
                "screenshot": f"/falls/{fall_count}/screenshot",
                "clip": f"/falls/{fall_count}/clip",
                "_clip_path": video_path,
                "_screenshot_path": screenshot_path,
                "_json_path": jf,
            }

            with fall_events_lock:
                fall_events.append(ui_evt)
            push_sse({"type": "fall_detected", "fall": ui_evt})


def watch_recoveries():
    """Watch recoveries.txt for new recovery lines and push SSE events."""
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
                            push_sse({"type": "fall_recovered", "fall": evt})
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
                push_sse({"type": "fall_enriched", "fall": evt})


# ═══════════════════════════════════════════════════════════════════
# DETECTION THREAD
# ═══════════════════════════════════════════════════════════════════

def detection_thread():
    if os.path.isfile(RECOVERIES_FILE):
        os.remove(RECOVERIES_FILE)
    run_fall_detection_live(0, output_filename=OUTPUT_STEM + ".mp4")


# ═══════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with ui_frame_lock:
                frame = ui_frame_jpeg
            if frame is None:
                time.sleep(0.03)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/events")
def events():
    q = Queue(maxsize=200)
    with sse_lock:
        sse_queues.append(q)

    with fall_events_lock:
        for evt in fall_events:
            q.put(json.dumps({"type": "fall_history", "fall": evt}))

    def stream():
        try:
            while True:
                try:
                    payload = q.get(timeout=30)
                    yield f"data: {payload}\n\n"
                except Empty:
                    yield ": keepalive\n\n"
        finally:
            with sse_lock:
                if q in sse_queues:
                    sse_queues.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/falls")
def list_falls():
    with fall_events_lock:
        safe = [{k: v for k, v in e.items() if not k.startswith("_")} for e in fall_events]
    return jsonify(safe)


@app.route("/falls/<int:fall_id>/screenshot")
def fall_screenshot(fall_id):
    with fall_events_lock:
        for evt in fall_events:
            if evt["id"] == fall_id and evt.get("_screenshot_path"):
                return send_file(evt["_screenshot_path"], mimetype="image/jpeg")
    return "Not found", 404


@app.route("/falls/<int:fall_id>/clip")
def fall_clip(fall_id):
    with fall_events_lock:
        for evt in fall_events:
            if evt["id"] == fall_id and evt.get("_clip_path"):
                return send_file(evt["_clip_path"], mimetype="video/mp4")
    return "Not found", 404


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=detection_thread, daemon=True).start()
    threading.Thread(target=watch_falls, daemon=True).start()
    threading.Thread(target=watch_recoveries, daemon=True).start()
    threading.Thread(target=watch_enrichment, daemon=True).start()

    print("\n  Vigilens UI running at http://localhost:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
