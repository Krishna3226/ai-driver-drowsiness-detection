"""
=============================================================
 AI Driver Monitoring System
 DAY 6-8 DEMO: State Machine + Audio + Logging
=============================================================
This file combines:
- Day 6: Unified temporal state machine
- Day 7: HUD polish and audio alert integration
- Day 8: Session logging and summary export
=============================================================
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import queue

try:
    import winsound
except ImportError:
    winsound = None


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]
LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95, 78,
]

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

EAR_THRESHOLD = 0.18
MICROSLEEP_THRESHOLD = 0.10
DROWSY_SECONDS = 1.5

MAR_THRESHOLD = 0.60
YAWN_SECONDS = 3.0
ABSENT_SECONDS = 1.0

YAW_THRESHOLD = 32.0
PITCH_DOWN_THRESHOLD = 15.0
ROLL_SIDE_THRESHOLD = 75.0
HEAD_DOWN_ROLL_THRESHOLD = 85.0
HEAD_DOWN_PITCH_ABS_LIMIT = 35.0
HEAD_DOWN_YAW_MIN = 15.0
HEAD_POSE_ALERT_SECONDS = 1.5
POSE_BASELINE_SAMPLES = 30
DISTRACTED_SECONDS = HEAD_POSE_ALERT_SECONDS
FACE_REACQUIRE_SECONDS = 0.30

# Pose hysteresis thresholds (enter vs exit) to reduce label flicker.
YAW_THRESHOLD_EXIT = 24.0
PITCH_DOWN_THRESHOLD_EXIT = 10.0
ROLL_SIDE_THRESHOLD_EXIT = 60.0
HEAD_DOWN_ROLL_THRESHOLD_EXIT = 72.0

SIDE_CONFIRM_SECONDS = 0.25
SIDE_SWITCH_SECONDS = 0.45
TRACKING_LOSS_GRACE_SECONDS = 0.45
YAW_BIAS_ADAPT_ALPHA = 0.03

# Camera orientation compensation (common webcam mirror/sign mismatch).
INVERT_YAW = True
DOWN_IS_POSITIVE_PITCH = True

STATE_NORMAL = 0
STATE_YAWNING = 1
STATE_DROWSY = 2
STATE_DISTRACTED = 3
STATE_ABSENT = 4

LOG_SNAPSHOT_SECONDS = 1.0
ALERT_SOUND_COOLDOWN = 2.0

LIGHT_GREEN = (120, 255, 120)
LIGHT_GREEN_BOX = (70, 255, 140)
NORMAL_GREEN = (0, 220, 120)
WARNING_ORANGE = (0, 165, 255)
CRITICAL_RED = (0, 60, 255)
HEAD_VALUE_COLOR = (255, 220, 80)
POSE_HOLD_COLOR = (255, 190, 80)
PANEL_BG = (18, 18, 34)

SESSION_LOG_DIR = Path(__file__).resolve().parent / "session_logs"
SESSION_LOG_DIR.mkdir(exist_ok=True)


class CaptureThread(threading.Thread):
    def __init__(self, src=0, queue_size=2):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {src}")
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = True
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue

            frame = cv2.resize(frame, (640, 480))
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def get_frame(self):
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.cap.release()


class SessionLogger:
    def __init__(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = SESSION_LOG_DIR / f"day8_session_{stamp}.csv"
        self.rows = []
        self.last_snapshot_time = 0.0
        self.last_signature = None

    def record(self, payload):
        now = time.time()
        absent_duration = payload.get("absent_duration")
        if absent_duration is None:
            # Fallback for log payloads that store absence as formatted timer text.
            absent_duration = payload.get("absent_timer", 0.0)
        try:
            absent_duration = float(absent_duration)
        except (TypeError, ValueError):
            absent_duration = 0.0

        signature = (
            payload["state_id"],
            payload["status_text"],
            payload["face_detected"],
            payload["drowsy_alert"],
            payload["yawn_alert"],
            payload["distracted_alert"],
            absent_duration >= ABSENT_SECONDS,
        )
        should_snapshot = (
            signature != self.last_signature
            or now - self.last_snapshot_time >= LOG_SNAPSHOT_SECONDS
        )
        if should_snapshot:
            self.rows.append(payload.copy())
            self.last_signature = signature
            self.last_snapshot_time = now

    def save(self):
        if not self.rows:
            return None

        fieldnames = [
            "timestamp",
            "state_id",
            "status_text",
            "face_detected",
            "ear",
            "mar",
            "pitch",
            "yaw",
            "roll",
            "eye_state",
            "mouth_state",
            "pose_state",
            "ear_timer",
            "mar_timer",
            "pose_timer",
            "absent_timer",
            "drowsy_alert",
            "yawn_alert",
            "distracted_alert",
        ]

        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        return self.path


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)


def lm_to_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [lm_to_px(landmarks[idx], w, h) for idx in eye_indices]
    vertical_1 = euclidean_distance(pts[1], pts[5])
    vertical_2 = euclidean_distance(pts[2], pts[4])
    horizontal = euclidean_distance(pts[0], pts[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks, w, h):
    top_lip = lm_to_px(landmarks[MOUTH[0]], w, h)
    bottom_lip = lm_to_px(landmarks[MOUTH[1]], w, h)
    left_corner = lm_to_px(landmarks[MOUTH[2]], w, h)
    right_corner = lm_to_px(landmarks[MOUTH[3]], w, h)
    vertical = euclidean_distance(top_lip, bottom_lip)
    horizontal = euclidean_distance(left_corner, right_corner)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def angle_delta(current, baseline):
    delta = current - baseline
    if delta > 180:
        delta -= 360
    elif delta < -180:
        delta += 360
    return delta


def estimate_head_pose(landmarks, w, h):
    # 1. Capture 2D landmarks from screen pixel coordinates
    image_points = np.array(
        [
            lm_to_px(landmarks[NOSE_TIP], w, h),
            lm_to_px(landmarks[CHIN], w, h),
            lm_to_px(landmarks[LEFT_EYE_OUTER], w, h),
            lm_to_px(landmarks[RIGHT_EYE_OUTER], w, h),
            lm_to_px(landmarks[LEFT_MOUTH_CORNER], w, h),
            lm_to_px(landmarks[RIGHT_MOUTH_CORNER], w, h),
        ],
        dtype=np.float64,
    )

    # 2. Realigned 3D Generic Anthropometric Model Point Array
    # Matches screen coordinates: +X is right, +Y is down, +Z is out toward camera
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),             # Nose tip reference origin
            (0.0, 330.0, -65.0),         # Chin (Below nose = Positive Y)
            (-225.0, -170.0, -135.0),    # Left eye outer corner (Above nose = Negative Y)
            (225.0, -170.0, -135.0),     # Right eye outer corner (Above nose = Negative Y)
            (-150.0, 150.0, -125.0),     # Left mouth corner (Below nose = Positive Y)
            (150.0, 150.0, -125.0),      # Right mouth corner (Below nose = Positive Y)
        ],
        dtype=np.float64,
    )

    focal_length = w
    camera_matrix = np.array(
        [
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    # Convert rotation vector to matrix then to stable Tait-Bryan angles.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        raw_pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        raw_yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        raw_roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        raw_pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        raw_yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        raw_roll = 0.0

    # Coordinate convention used in this file:
    # Pitch: Negative = head down, Positive = head up.
    # Yaw: Negative = left, Positive = right.
    pitch = -raw_pitch
    yaw = raw_yaw
    roll = raw_roll

    # Project the 3D visual pointer nose line straight out along the +Z axis
    nose_2d = image_points[0]
    nose_3d_projection, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 300.0)], dtype=np.float64), # Scaled down for best HUD look
        rotation_vec,
        translation_vec,
        camera_matrix,
        dist_coeffs,
    )
    nose_end = tuple(nose_3d_projection[0][0].astype(int))

    return pitch, yaw, roll, tuple(nose_2d.astype(int)), nose_end



def classify_head_pose(pitch, yaw, roll, prev_state="FORWARD", stable_side="FORWARD"):
    # Use pitch as primary head-down signal, but suppress head-down when the
    # face is strongly turned sideways (common source of false head-down labels).
    side_heavy = abs(yaw) > (YAW_THRESHOLD + 16)
    pitch_value = pitch if DOWN_IS_POSITIVE_PITCH else -pitch
    if prev_state == "HEAD DOWN":
        head_down_active = (pitch_value > PITCH_DOWN_THRESHOLD_EXIT) and not side_heavy
    else:
        head_down_active = (pitch_value > PITCH_DOWN_THRESHOLD) and not side_heavy

    if head_down_active:
        return "HEAD DOWN", CRITICAL_RED
        
    if stable_side == "LOOKING LEFT":
        return "LOOKING LEFT", WARNING_ORANGE
    if stable_side == "LOOKING RIGHT":
        return "LOOKING RIGHT", WARNING_ORANGE
        
    return "FORWARD", NORMAL_GREEN


def draw_landmark_group(frame, landmarks, indices, w, h, color, label=None):
    pts = []
    for idx in indices:
        x, y = lm_to_px(landmarks[idx], w, h)
        cv2.circle(frame, (x, y), 2, color, -1)
        pts.append((x, y))

    if label and pts:
        cx = sum(p[0] for p in pts) // len(pts)
        cy = min(p[1] for p in pts) - 8
        cv2.putText(
            frame,
            label,
            (cx - 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return pts


def draw_bbox(frame, pts, color, padding=4):
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cv2.rectangle(
        frame,
        (min(xs) - padding, min(ys) - padding),
        (max(xs) + padding, max(ys) + padding),
        color,
        1,
    )


def draw_top_hud(frame, fps, w):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(
        frame,
        "AI DRIVER MONITORING - DAY 6/7/8 DEMO",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    fps_color = NORMAL_GREEN if fps >= 20 else WARNING_ORANGE
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 110, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        fps_color,
        1,
        cv2.LINE_AA,
    )


def draw_info_panel(frame, lines, w, anchor_right=True):
    panel_w = 220
    margin = 8
    panel_x = w - panel_w - margin if anchor_right else margin + 8
    panel_y = 55
    line_h = 17
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x - 8, panel_y - 18),
        (panel_x + panel_w, panel_y + len(lines) * line_h + 6),
        PANEL_BG,
        -1,
    )
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(
            frame,
            txt,
            (panel_x, panel_y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            col,
            1,
            cv2.LINE_AA,
        )


def draw_bottom_status(frame, text, color, w, h, state_id):
    overlay = frame.copy()
    if state_id in (STATE_DROWSY, STATE_ABSENT):
        bar_color = (20, 20, 140)
    elif state_id in (STATE_YAWNING, STATE_DISTRACTED):
        bar_color = (20, 80, 130)
    else:
        bar_color = (20, 70, 20)

    cv2.rectangle(overlay, (0, h - 34), (w, h), bar_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(
        frame,
        text,
        (10, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"STATE ID: {state_id}",
        (w - 130, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def update_timer(active, start_time):
    if active:
        if start_time is None:
            start_time = time.time()
        return start_time, time.time() - start_time
    return None, 0.0


def resolve_state(metrics):
    if metrics["absent_duration"] >= ABSENT_SECONDS:
        return STATE_ABSENT, "OPERATOR ABSENT - INTERLOCK", CRITICAL_RED
    if metrics["drowsy_alert"]:
        return STATE_DROWSY, "DROWSY ALERT", CRITICAL_RED
    if metrics["yawn_alert"]:
        return STATE_YAWNING, "YAWNING DETECTED", WARNING_ORANGE
    if metrics["distracted_alert"]:
        return STATE_DISTRACTED, "DISTRACTED HEAD POSE", WARNING_ORANGE
    return STATE_NORMAL, "ALERT / NORMAL", NORMAL_GREEN


audio_lock = threading.Lock()
last_audio_time = 0.0


def _beep_pattern(state_id):
    if winsound is None:
        print("\a", end="", flush=True)
        return

    if state_id == STATE_ABSENT:
        pattern = [(1100, 120), (900, 120), (700, 180)]
    elif state_id == STATE_DROWSY:
        pattern = [(1000, 160), (1000, 160)]
    elif state_id == STATE_YAWNING:
        pattern = [(800, 120)]
    elif state_id == STATE_DISTRACTED:
        pattern = [(900, 120), (750, 120)]
    else:
        return

    for freq, dur in pattern:
        try:
            winsound.Beep(freq, dur)
        except Exception:
            break


def trigger_audio_alert(state_id):
    global last_audio_time
    now = time.time()
    with audio_lock:
        if now - last_audio_time < ALERT_SOUND_COOLDOWN:
            return
        last_audio_time = now

    thread = threading.Thread(target=_beep_pattern, args=(state_id,), daemon=True)
    thread.start()


def main():
    print("=" * 60)
    print(" AI Driver Monitoring System - Day 6/7/8 Demo")
    print(" State machine + audio + logging")
    print("=" * 60)
    print(" Controls:")
    print("   Q - quit")
    print("   M - toggle full mesh overlay")
    print("   C - recalibrate straight head pose")
    print("   F - toggle fullscreen")
    print("   I - toggle yaw inversion")
    print("=" * 60)

    window_name = "AI Driver Monitoring - Day 6/7/8 Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    is_fullscreen = True
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    capture = CaptureThread(src=0)
    capture.start()

    logger = SessionLogger()

    show_mesh = False
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0

    ear_start = None
    mar_start = None
    distract_start = None
    absent_start = None
    face_reacquire_start = None
    pose_baseline = None
    pose_baseline_samples = []
    pose_label_state = "FORWARD"
    stable_side = "FORWARD"
    side_candidate = None
    side_candidate_start = None
    smoothed_pitch = None
    smoothed_yaw = None
    smoothed_roll = None
    tracking_loss_start = None
    pose_smoothing_alpha = 0.25
    last_state_id = None
    invert_yaw = INVERT_YAW
    yaw_bias = 0.0

    while True:
        frame = capture.get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        face_detected = result.multi_face_landmarks is not None
        face_center_x = None
        face_min_x = None
        face_max_x = None

        avg_ear = None
        mar = None
        pitch = yaw = roll = None
        eye_state = "NO FACE"
        mouth_state = "NO FACE"
        pose_state = "NO FACE"
        closed_duration = 0.0
        yawn_duration = 0.0
        absent_duration = 0.0
        head_pose_duration = 0.0
        drowsy_alert = False
        yawn_alert = False
        distracted_alert = False
        status_text = "NO FACE - CHECKING"
        status_color = WARNING_ORANGE

        if face_detected:
            if absent_start is not None:
                if face_reacquire_start is None:
                    face_reacquire_start = time.time()
                absent_duration = time.time() - absent_start
                if time.time() - face_reacquire_start >= FACE_REACQUIRE_SECONDS:
                    absent_start = None
                    face_reacquire_start = None
                    absent_duration = 0.0
            else:
                face_reacquire_start = None
            lms = result.multi_face_landmarks[0].landmark
            face_center_x = np.mean([pt.x for pt in lms]) * w
            face_min_x = min(pt.x for pt in lms) * w
            face_max_x = max(pt.x for pt in lms) * w

            if show_mesh:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=result.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
                )

            le_pts = draw_landmark_group(frame, lms, LEFT_EYE, w, h, LIGHT_GREEN, "L-EYE")
            re_pts = draw_landmark_group(frame, lms, RIGHT_EYE, w, h, LIGHT_GREEN, "R-EYE")
            lip_pts = draw_landmark_group(frame, lms, LIPS, w, h, WARNING_ORANGE, "MOUTH")
            key_mouth_pts = draw_landmark_group(frame, lms, MOUTH, w, h, (0, 255, 255))
            draw_bbox(frame, le_pts, LIGHT_GREEN_BOX)
            draw_bbox(frame, re_pts, LIGHT_GREEN_BOX)
            draw_bbox(frame, lip_pts, (0, 140, 220))

            if len(key_mouth_pts) == 4:
                cv2.line(frame, key_mouth_pts[0], key_mouth_pts[1], (0, 255, 255), 1)
                cv2.line(frame, key_mouth_pts[2], key_mouth_pts[3], (0, 255, 255), 1)

            avg_ear = (
                eye_aspect_ratio(lms, LEFT_EYE, w, h)
                + eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            ) / 2.0
            mar = mouth_aspect_ratio(lms, w, h)

            if avg_ear < EAR_THRESHOLD:
                ear_start, closed_duration = update_timer(True, ear_start)
                drowsy_alert = closed_duration >= DROWSY_SECONDS or avg_ear < MICROSLEEP_THRESHOLD
                eye_state = "DROWSY" if drowsy_alert else "EYES CLOSED"
            else:
                ear_start = None
                drowsy_alert = False
                eye_state = "EYES OPEN"

            if mar > MAR_THRESHOLD:
                mar_start, yawn_duration = update_timer(True, mar_start)
                yawn_alert = yawn_duration >= YAWN_SECONDS
                mouth_state = "YAWNING" if yawn_alert else "MOUTH OPEN"
            else:
                mar_start = None
                yawn_alert = False
                mouth_state = "NORMAL"

            pose = estimate_head_pose(lms, w, h)
            if pose is not None:
                raw_pitch, raw_yaw, raw_roll, nose_start, nose_end = pose
                cv2.line(frame, nose_start, nose_end, (255, 220, 80), 2)
                cv2.circle(frame, nose_start, 3, (200, 80, 255), -1)

                if pose_baseline is None:
                    pose_baseline_samples.append((raw_pitch, raw_yaw, raw_roll))
                    pose_state = f"CALIBRATING {len(pose_baseline_samples)}/{POSE_BASELINE_SAMPLES}"
                    if len(pose_baseline_samples) >= POSE_BASELINE_SAMPLES:
                        pose_baseline = np.mean(np.array(pose_baseline_samples), axis=0)
                        pose_baseline_samples = []
                        pitch = yaw = roll = 0.0
                        pose_state = "FORWARD"
                        yaw_bias = 0.0
                        pose_label_state = "FORWARD"
                        stable_side = "FORWARD"
                        side_candidate = None
                        side_candidate_start = None
                else:
                    pitch = angle_delta(raw_pitch, pose_baseline[0])
                    yaw = angle_delta(raw_yaw, pose_baseline[1])
                    roll = angle_delta(raw_roll, pose_baseline[2])

                    # Smooth pose values to reduce one-frame spikes and false alerts.
                    if smoothed_pitch is None:
                        smoothed_pitch, smoothed_yaw, smoothed_roll = pitch, yaw, roll
                    else:
                        smoothed_pitch = (1 - pose_smoothing_alpha) * smoothed_pitch + pose_smoothing_alpha * pitch
                        smoothed_yaw = (1 - pose_smoothing_alpha) * smoothed_yaw + pose_smoothing_alpha * yaw
                        smoothed_roll = (1 - pose_smoothing_alpha) * smoothed_roll + pose_smoothing_alpha * roll

                    pitch, yaw, roll = smoothed_pitch, smoothed_yaw, smoothed_roll
                    if invert_yaw:
                        yaw = -yaw
                    yaw_rel = yaw - yaw_bias

                    # Slowly adapt neutral yaw center only while confidently forward.
                    if (
                        pose_label_state == "FORWARD"
                        and abs(yaw_rel) <= YAW_THRESHOLD_EXIT
                        and abs(pitch) <= PITCH_DOWN_THRESHOLD_EXIT
                    ):
                        yaw_bias = (1.0 - YAW_BIAS_ADAPT_ALPHA) * yaw_bias + YAW_BIAS_ADAPT_ALPHA * yaw

                    # Stabilize left/right by using yaw-only direction and time confirmation.
                    if yaw_rel < -YAW_THRESHOLD:
                        side_target = "LOOKING LEFT"
                    elif yaw_rel > YAW_THRESHOLD:
                        side_target = "LOOKING RIGHT"
                    elif abs(yaw_rel) <= YAW_THRESHOLD_EXIT:
                        side_target = "FORWARD"
                    else:
                        side_target = stable_side

                    if side_target != stable_side:
                        if side_candidate != side_target:
                            side_candidate = side_target
                            side_candidate_start = time.time()
                        else:
                            between_sides = (
                                stable_side in ("LOOKING LEFT", "LOOKING RIGHT")
                                and side_target in ("LOOKING LEFT", "LOOKING RIGHT")
                                and stable_side != side_target
                            )
                            hold_needed = SIDE_SWITCH_SECONDS if between_sides else SIDE_CONFIRM_SECONDS
                            
                            if side_candidate_start is not None:
                                if time.time() - side_candidate_start >= hold_needed:
                                    stable_side = side_target
                                    side_candidate = None
                                    side_candidate_start = None
                    else:
                        side_candidate = None
                        side_candidate_start = None

                    pose_state, _ = classify_head_pose(
                        pitch,
                        yaw_rel,
                        roll,
                        pose_label_state,
                        stable_side,
                    )
                    pose_label_state = pose_state
                    yaw = yaw_rel

                    # Keep distracted-state timer aligned with the stabilized
                    # pose label to prevent "DISTRACTED: FORWARD" mismatches.
                    distracted_condition = pose_state != "FORWARD"
                    distract_start, head_pose_duration = update_timer(distracted_condition, distract_start)
                    distracted_alert = head_pose_duration >= DISTRACTED_SECONDS
                    tracking_loss_start = None
            else:
                if tracking_loss_start is None:
                    tracking_loss_start = time.time()
                if time.time() - tracking_loss_start >= TRACKING_LOSS_GRACE_SECONDS:
                    distract_start = None
                    pose_label_state = "FORWARD"
                    stable_side = "FORWARD"
                    side_candidate = None
                    side_candidate_start = None
                    smoothed_pitch = None
                    smoothed_yaw = None
                    smoothed_roll = None
                else:
                    # Hold last stable pose briefly to avoid losing distracted state
                    # at extreme yaw angles where landmark tracking can flicker.
                    pose_state = pose_label_state
                    pitch = smoothed_pitch
                    yaw = smoothed_yaw
                    roll = smoothed_roll
                    distracted_condition = pose_state != "FORWARD"
                    distract_start, head_pose_duration = update_timer(distracted_condition, distract_start)
                    distracted_alert = head_pose_duration >= DISTRACTED_SECONDS

            if pose_state != "FORWARD" and not pose_state.startswith("CALIBRATING"):
                # Keep eye-timer history so true drowsiness is not suppressed
                # when the driver is also looking away/head-down.
                if avg_ear is not None and avg_ear < EAR_THRESHOLD:
                    eye_state = "POSE CHECK"

            if drowsy_alert:
                status_text = "DROWSY ALERT"
                status_color = CRITICAL_RED
            elif yawn_alert:
                status_text = "YAWNING DETECTED"
                status_color = WARNING_ORANGE
            elif distracted_alert:
                status_text = f"DISTRACTED: {pose_state}"
                status_color = WARNING_ORANGE
            elif pose_state not in ("FORWARD", "NO FACE") and not pose_state.startswith("CALIBRATING"):
                status_text = f"HEAD POSE: {pose_state}"
                status_color = WARNING_ORANGE if pose_state != "HEAD DOWN" else CRITICAL_RED
            else:
                status_text = "ALERT / NORMAL"
                status_color = NORMAL_GREEN
        else:
            if absent_start is None:
                absent_start = time.time()
            absent_duration = time.time() - absent_start

            if tracking_loss_start is None:
                tracking_loss_start = time.time()
            tracking_loss_elapsed = time.time() - tracking_loss_start

            if tracking_loss_elapsed >= TRACKING_LOSS_GRACE_SECONDS:
                face_reacquire_start = None
                ear_start = None
                mar_start = None
                distract_start = None
                pose_label_state = "FORWARD"
                stable_side = "FORWARD"
                side_candidate = None
                side_candidate_start = None
                smoothed_pitch = None
                smoothed_yaw = None
                smoothed_roll = None
                drowsy_alert = False
                yawn_alert = False
                distracted_alert = False
                if absent_duration >= ABSENT_SECONDS:
                    status_text = "OPERATOR ABSENT - INTERLOCK"
                    status_color = CRITICAL_RED
            else:
                # Brief no-face dropout: keep previous pose/distracted state alive.
                pose_state = pose_label_state
                pitch = smoothed_pitch
                yaw = smoothed_yaw
                roll = smoothed_roll
                distracted_condition = pose_state != "FORWARD"
                distract_start, head_pose_duration = update_timer(distracted_condition, distract_start)
                distracted_alert = head_pose_duration >= DISTRACTED_SECONDS

        metrics = {
            "absent_duration": absent_duration,
            "drowsy_alert": drowsy_alert,
            "yawn_alert": yawn_alert,
            "distracted_alert": distracted_alert,
        }
        state_id, resolved_text, resolved_color = resolve_state(metrics)

        if status_text in ("ALERT / NORMAL", "NO FACE - CHECKING") or status_color == NORMAL_GREEN:
            status_text = resolved_text
            status_color = resolved_color
        elif state_id == STATE_ABSENT:
            status_text = resolved_text
            status_color = resolved_color

        if state_id in (STATE_DROWSY, STATE_YAWNING, STATE_DISTRACTED, STATE_ABSENT):
            if last_state_id != state_id:
                trigger_audio_alert(state_id)
        last_state_id = state_id

        logger.record(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "state_id": state_id,
                "status_text": status_text,
                "face_detected": face_detected,
                "ear": f"{avg_ear:.3f}" if avg_ear is not None else "",
                "mar": f"{mar:.3f}" if mar is not None else "",
                "pitch": f"{pitch:.2f}" if pitch is not None else "",
                "yaw": f"{yaw:.2f}" if yaw is not None else "",
                "roll": f"{roll:.2f}" if roll is not None else "",
                "eye_state": eye_state,
                "mouth_state": mouth_state,
                "pose_state": pose_state,
                "ear_timer": f"{closed_duration:.2f}",
                "mar_timer": f"{yawn_duration:.2f}",
                "pose_timer": f"{head_pose_duration:.2f}",
                "absent_timer": f"{absent_duration:.2f}",
                "drowsy_alert": drowsy_alert,
                "yawn_alert": yawn_alert,
                "distracted_alert": distracted_alert,
            }
        )

        lines = [
            ("DAY 6: STATE MACHINE", (200, 200, 200)),
            ("DAY 7: AUDIO + HUD", (180, 220, 255)),
            ("DAY 8: SESSION LOG", (180, 255, 200)),
            (
                "Face: detected" if face_detected else "Face: not detected",
                NORMAL_GREEN if face_detected else WARNING_ORANGE,
            ),
            (
                f"EAR: {avg_ear:.2f}" if avg_ear is not None else "EAR: --",
                NORMAL_GREEN if avg_ear is not None and avg_ear >= EAR_THRESHOLD else WARNING_ORANGE,
            ),
            (
                f"MAR: {mar:.2f}" if mar is not None else "MAR: --",
                NORMAL_GREEN if mar is not None and mar <= MAR_THRESHOLD else WARNING_ORANGE,
            ),
            (f"Pitch: {pitch:.1f}" if pitch is not None else "Pitch: --", HEAD_VALUE_COLOR),
            (f"Yaw: {yaw:.1f}" if yaw is not None else "Yaw: --", HEAD_VALUE_COLOR),
            (f"Roll: {roll:.1f}" if roll is not None else "Roll: --", HEAD_VALUE_COLOR),
            (f"Eyes: {eye_state}", NORMAL_GREEN if eye_state == "EYES OPEN" else WARNING_ORANGE),
            (f"Mouth: {mouth_state}", NORMAL_GREEN if mouth_state == "NORMAL" else WARNING_ORANGE),
            (f"Pose: {pose_state}", WARNING_ORANGE if pose_state != "FORWARD" else NORMAL_GREEN),
            (f"Eye timer: {closed_duration:.1f}s", (150, 150, 150)),
            (f"Yawn timer: {yawn_duration:.1f}s", (150, 150, 150)),
            (f"Pose timer: {head_pose_duration:.1f}s", (150, 150, 150)),
            (f"Absent: {absent_duration:.1f}s", (150, 150, 150)),
            (f"State ID: {state_id}", status_color),
        ]

        # Keep HUD away from the face by anchoring panel on lower-overlap side.
        anchor_right = True
        panel_w = 220
        if face_min_x is not None and face_max_x is not None:
            left_panel_l, left_panel_r = 0, panel_w
            right_panel_l, right_panel_r = w - panel_w, w
            overlap_left = max(0.0, min(face_max_x, left_panel_r) - max(face_min_x, left_panel_l))
            overlap_right = max(0.0, min(face_max_x, right_panel_r) - max(face_min_x, right_panel_l))
            anchor_right = overlap_right <= overlap_left
        elif face_center_x is not None:
            anchor_right = face_center_x < (w * 0.52)

        draw_top_hud(frame, fps_display, w)
        draw_info_panel(frame, lines, w, anchor_right=anchor_right)
        draw_bottom_status(frame, status_text, status_color, w, h, state_id)

        if state_id != STATE_NORMAL:
            cv2.putText(
                frame,
                status_text,
                (w // 2 - 180, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                status_color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mesh = not show_mesh
            print(f"Mesh overlay: {'ON' if show_mesh else 'OFF'}")
        if key == ord("c"):
            pose_baseline = None
            pose_baseline_samples = []
            distract_start = None
            yaw_bias = 0.0
            pose_label_state = "FORWARD"
            stable_side = "FORWARD"
            side_candidate = None
            side_candidate_start = None
            smoothed_pitch = None
            smoothed_yaw = None
            smoothed_roll = None
            print("Head pose calibration reset. Face forward for a second.")
        if key == ord("f"):
            is_fullscreen = not is_fullscreen
            mode = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
        if key == ord("i"):
            invert_yaw = not invert_yaw
            print(f"Yaw inversion: {'ON' if invert_yaw else 'OFF'}")

    capture.stop()
    face_mesh.close()
    cv2.destroyAllWindows()

    log_path = logger.save()
    if log_path is not None:
        print(f"Session log saved: {log_path}")
    print("Session ended.")


if __name__ == "__main__":
    main()
