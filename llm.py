"""
Fall Incident Summary Generator
================================
Uses the Hugging Face Inference API to generate a structured clinical summary
from a fall detection JSON file.

Setup:
    pip install huggingface_hub

Usage:
    python fall_summary_generator.py --json_path 1_0.json
    python fall_summary_generator.py --json_path 1_0.json --model "mistralai/Mistral-7B-Instruct-v0.3"
"""
import json
import argparse
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
JSON_PATH = "1_0.json"
# Other good options:
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#   "HuggingFaceH4/zephyr-7b-beta"

load_dotenv()  # reads .env file automatically
HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── JSON → compact data extraction ────────────────────────────────────────────

def extract_features(data: dict) -> dict:
    """Pull the key numbers and facts out of the raw fall JSON."""
    frames = data.get("frames", [])
    foi = data.get("fall_object_interaction", {})
    summary = foi.get("summary", {})
    ffc = foi.get("first_floor_contact", {})
    obj_interactions = foi.get("object_interactions", [])

    # Fall duration
    t_start = float(data["fall_start"][1])
    t_end   = float(data["fall_end"][1])
    duration_sec = round(t_end - t_start, 2)

    # Angle range during fall
    angles = [float(f["angle"]) for f in frames]
    angle_min, angle_max = round(min(angles), 1), round(max(angles), 1)

    # Frame where fall_ongoing flips True
    transition_frames = [
        f["frame_id"] for f in frames if f.get("fall_ongoing") == "True"
    ]
    first_fall_frame = transition_frames[0] if transition_frames else data["fall_frame"]

    # Head speed / accel
    head_speed = float(data.get("head_speed", 0))
    head_accel = float(data.get("head_accel", 0))
    shoulder_speed = float(data.get("shoulder_speed", 0))

    # Final frame posture
    last_frame = frames[-1] if frames else {}
    final_horizontal = last_frame.get("horizontal", "False") == "True"

    # Object interactions
    high_risk_objects = [
        o for o in obj_interactions if o.get("injury_risk") == "HIGH"
    ]

    return {
        "fall_id":          data.get("fall_id", "N/A"),
        "person_id":        data.get("person_id", "N/A"),
        "pre_fall_posture": foi.get("pre_fall_posture", summary.get("pre_fall_posture", "unknown")),
        "scene":            data.get("scene_context", {}).get("Indoor/Outdoor", "unknown"),
        "fall_detected":    summary.get("fell", "True") == "True",
        "duration_sec":     duration_sec,
        "start_frame":      data["fall_start"][0],
        "end_frame":        data["fall_end"][0],
        "first_fall_frame": first_fall_frame,
        "angle_min":        angle_min,
        "angle_max":        angle_max,
        "head_speed":       round(head_speed, 2),
        "head_accel":       round(head_accel, 2),
        "shoulder_speed":   round(shoulder_speed, 2),
        "flat_fall":        ffc.get("flat_fall", "False") == "True",
        "floor_contact_part": ffc.get("body_part", "unknown"),
        "all_contact_parts":  ffc.get("all_parts", []),
        "final_horizontal": final_horizontal,
        "highest_risk_part": summary.get("highest_risk_body_part", "None"),
        "high_risk_objects": [o["object"] for o in high_risk_objects],
        "objects_after_fall": summary.get("objects_touching", {}).get("after_fall", []),
        "falling_objects":   summary.get("falling_objects_during_fall", []),
        "was_sitting_on":    summary.get("was_sitting_on", []),
    }


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(features: dict) -> str:
    contact_parts = ', '.join(features['all_contact_parts']) if features['all_contact_parts'] else 'none'
    flat = features['flat_fall']
    h_risk = features['highest_risk_part']
    objs = ', '.join(features['high_risk_objects']) if features['high_risk_objects'] else None

    chair_note = "Chair present near patient post-fall — assess for secondary injury" if objs else "No hazardous objects in contact with patient"
    risk_note = ("No single high-risk body part flagged — assess all contact points" 
                 if h_risk == 'None' else f"High-risk body part: {h_risk} — prioritize assessment")
    contact_type = "Full-body flat impact" if flat else "Partial body contact"
    head_note = "No head-first impact" if features['floor_contact_part'] != 'head' else "Head-first impact detected — high concussion risk"
    fall_detected = "Yes" if features['fall_detected'] else "No"

    return f"""You are writing a fall incident report for an EMS/paramedic crew. Be concise and clinical. Only include what matters for patient assessment and triage.

STRICT RULES:
- No markdown, no bold, no asterisks for emphasis, no dash dividers.
- Use only "* " to start each bullet point.
- Do not mention frame numbers, pixel values, or any camera/system data.
- Write as if describing what was observed on scene by a first responder.
- Stop after the last bullet. No closing remarks.

OUTPUT EXACTLY THIS FORMAT:

Incident ID: {features['fall_id']}
Person ID: {features['person_id']}
Pre-fall posture: {features['pre_fall_posture'].capitalize()}
Fall detected: {fall_detected}
Fall duration: ~{features['duration_sec']} seconds
Scene: {features['scene'].capitalize()}

Mechanism of fall:
* Patient was standing when loss of balance occurred
* Rapid uncontrolled descent — body rotated from fully upright to ground-level without bracing
* No stumble, grab, or recovery attempt detected prior to impact
* Patient became fully horizontal during the fall

Impact details:
* Ground contact: {contact_type} — {contact_parts} contacted the floor
* {head_note}
* {chair_note}
* No objects knocked over during the fall

Severity assessment:
* Fall severity: HIGH — rapid descent with significant rotational force recorded
* {risk_note}
* Recommend spinal precautions given mechanism and speed of descent

Post-fall status:
* Patient remained on ground — no self-recovery observed
* Patient was horizontal at end of recording — duration on ground unknown
* Assess for hip fracture, wrist injury, and soft tissue trauma at all contact points

Write only the report. Stop after the last bullet."""



# ── LLM call ──────────────────────────────────────────────────────────────────

def generate_summary(prompt: str, model: str, token: str) -> str:
    import os
    hf_token = token if token != "hf_YOUR_TOKEN_HERE" else os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError(
            "No HuggingFace token found. Either set HF_TOKEN env var "
            "or paste your token into the HF_TOKEN variable in this script."
        )

    client = InferenceClient(model=model, token=hf_token)

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate fall incident summary from JSON")
    parser.add_argument("--json_path", default=JSON_PATH, required=True, help="Path to fall JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--output", default=None, help="Optional output .txt file path")
    args = parser.parse_args()

    # Load JSON
    with open(args.json_path, "r") as f:
        data = json.load(f)

    print(f"✅ Loaded fall JSON: {args.json_path}")

    # Extract features
    features = extract_features(data)
    print(f"✅ Extracted features for Person {features['person_id']}, Fall {features['fall_id']}")

    # Build prompt
    prompt = build_prompt(features)

    # Generate summary
    print(f"⏳ Calling model: {args.model} ...")
    summary = generate_summary(prompt, model=args.model, token=HF_TOKEN)

    print("\n" + "="*60)
    print(summary)
    print("="*60)

    # Optionally save
    if args.output:
        with open(args.output, "w") as f:
            f.write(summary)
        print(f"\n✅ Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
