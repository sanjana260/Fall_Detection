import argparse
import time

import cv2

from scene_context import SceneContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="0 for webcam, or path to a video file")
    parser.add_argument("--every_n", type=int, default=10, help="run scene model every N frames")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    ctx = SceneContext(device=args.device)
    ctx.reset()

    last_pred = None
    frame_i = 0
    t0 = time.perf_counter()
    fps = 0.0
    last_print_scene = None
    last_print_io = None


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1

        if frame_i % args.every_n == 0 or last_pred is None:
            start = time.perf_counter()
            last_pred = ctx.predict_bgr(frame, topk=args.topk)
            infer_ms = (time.perf_counter() - start) * 1000.0
            last_pred["infer_ms"] = infer_ms

            io = last_pred["indoor_outdoor"]
            scene = last_pred["scene_type"]

            # Print only if it changed (reduces spam). Comment this if you want always.
            if io != last_print_io or scene != last_print_scene:
                print(
                    f"[scene] io={last_pred['indoor_outdoor']}({last_pred['io_conf']:.2f}) "
                    f"scene={last_pred['scene_type']} raw={last_pred['scene_type_raw']}({last_pred['scene_conf_raw']:.2f}) "
                    f"top3={_fmt_topk(last_pred['topk'][:3])} "
                    f"device={last_pred['device']} infer_ms={infer_ms:.1f}"
                )
                last_print_io = io
                last_print_scene = scene

        # FPS estimate
        dt = time.perf_counter() - t0
        if dt >= 1.0:
            fps = frame_i / dt
            t0 = time.perf_counter()
            frame_i = 0

        # Overlay
        if last_pred is not None:
            text1 = f"{last_pred['indoor_outdoor']} ({last_pred['io_conf']:.2f})"
            text2 = f"{last_pred['scene_type']} ({last_pred['scene_conf']:.2f})"
            text3 = f"device={last_pred['device']} infer={last_pred.get('infer_ms', 0.0):.1f}ms FPS~{fps:.1f}"

            cv2.putText(frame, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, text3, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Scene Context", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break
        
    final_pred = ctx.final_video_prediction(topk=args.topk)
    print("[FINAL]", final_pred)

    cap.release()
    cv2.destroyAllWindows()

def _fmt_topk(topk):
    return ", ".join([f"{d['label']}:{d['prob']:.2f}" for d in topk])

if __name__ == "__main__":
    main()
