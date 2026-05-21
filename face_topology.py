"""
=============================================================
 AI Driver Monitoring System
 DAY 1–2: Async Webcam Thread + MediaPipe Face Topology
=============================================================
 Timeline reference: Days 1-2 — Environment & Topology
 - Async webcam thread (Thread 1)
 - 480p downscale → frame queue
 - MediaPipe 468-point face mesh
 - Map landmark indices for eyes and lips
 - HUD overlay: FPS counter, landmark dots, eye/mouth boxes
=============================================================
<<<<<<< HEAD
DAY3 
- EAR calculation from eye landmarks
- Clock based Drowsiness Alert when EAR stays LOW
=============================================================
=======
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f
"""

import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time

# ──────────────────────────────────────────────────────────
# MEDIAPIPE LANDMARK INDEX MAP
# These are the exact indices we will use for EAR/MAR later
# ──────────────────────────────────────────────────────────

# Left eye  (p1..p6 from Soukupova & Cech 2016)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
# Mouth (top lip, bottom lip, left corner, right corner)
MOUTH     = [13, 14, 78, 308]
# Full lip outline for MAR bounding box
LIPS      = [61, 146, 91, 181, 84, 17, 314, 405,
             321, 375, 291, 308, 324, 318, 402,
             317, 14, 87, 178, 88, 95, 78]
# Nose tip — used later for head pose (PnP)
NOSE_TIP  = 1

<<<<<<< HEAD
# EAR THRESHOLDS
EAR_THRESHOLD= 0.18
MICROSLEEP_THRESHOLD=0.10
DROWSY_SECONDS=1.5

=======
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f
# ──────────────────────────────────────────────────────────
# THREAD 1 — SENSOR INGESTION HANDLER
# Captures raw frames, downscales to 480p, pushes to queue
# ──────────────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    def __init__(self, src=0, queue_size=2):
        super().__init__(daemon=True)
        self.cap   = cv2.VideoCapture(src)
        self.queue = queue.Queue(maxsize=queue_size)
        self.running = True

        # Force 480p at source if camera supports it
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            # Downscale to 480p (ensures consistent resolution)
            frame = cv2.resize(frame, (640, 480))
            # Drop old frame if queue full — never block inference
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


# ──────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────────────────

mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces      = 1,
    refine_landmarks   = True,   # adds iris landmarks
    min_detection_confidence = 0.6,
    min_tracking_confidence  = 0.6,
)


# ──────────────────────────────────────────────────────────
# HELPER: pixel coords from normalised landmark
# ──────────────────────────────────────────────────────────

def lm_to_px(landmark, w, h):
    """Convert MediaPipe normalised landmark → (x, y) pixel tuple."""
    return int(landmark.x * w), int(landmark.y * h)

<<<<<<< HEAD
def euclidean_distance(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def eye_aspect_ratio(landmarks,eye_indices,w,h):
    """ Calculate EAR using 6 eye Landmarks"""
    pts=[lm_to_px(landmarks[idx],w,h) for idx in eye_indices]
    vertical_1=euclidean_distance(pts[1],pts[5])
    vertical_2=euclidean_distance(pts[2],pts[4])
    horizontal=euclidean_distance(pts[0],pts[3])

    if horizontal==0:
        return 0.0
    
    return(vertical_1+vertical_2)/(2.0*horizontal) 
=======
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f

# ──────────────────────────────────────────────────────────
# HELPER: draw a labelled landmark group
# ──────────────────────────────────────────────────────────

def draw_landmark_group(frame, landmarks, indices, w, h, color, label=None):
    pts = []
    for idx in indices:
        x, y = lm_to_px(landmarks[idx], w, h)
        cv2.circle(frame, (x, y), 2, color, -1)
        pts.append((x, y))
    if label and pts:
        cx = sum(p[0] for p in pts) // len(pts)
        cy = min(p[1] for p in pts) - 8
        cv2.putText(frame, label, (cx - 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return pts


# ──────────────────────────────────────────────────────────
# HELPER: bounding box around a set of points
# ──────────────────────────────────────────────────────────

def draw_bbox(frame, pts, color, padding=4):
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cv2.rectangle(frame,
                  (min(xs) - padding, min(ys) - padding),
                  (max(xs) + padding, max(ys) + padding),
                  color, 1)


# ──────────────────────────────────────────────────────────
# HUD OVERLAY
# ──────────────────────────────────────────────────────────

<<<<<<< HEAD
def draw_hud(frame, fps, face_detected, w, h,driver_state="NORMAL"):
=======
def draw_hud(frame, fps, face_detected, w, h):
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "AI DRIVER MONITORING", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS counter
    fps_color = (0, 220, 120) if fps >= 20 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 1, cv2.LINE_AA)

    # Face detection status
<<<<<<< HEAD

    # driver_state = 'NORMAL'
    if not face_detected:
        status_text = 'NO FACE — INTERLOCK'
        status_color = (0, 60, 220)
    elif driver_state == 'DROWSY':
        status_text='DROWSY ALERT'
        status_color=(0,165,220)
    elif driver_state == 'EYES CLOSED':
        status_text="EYES CLOSED"
        status_color=(0,165,255)
    else:
        status_text="FACE DETECTED"
        status_color=(0,220,120)
    overlay2=frame.copy()
    bar_color=(20,20,120) if status_text in ('NO FACE — INTERLOCK','DROWSY ALERT','EYES CLOSED ALERT') else (20,120,20)
    cv2.rectangle(overlay2,(0,h - 32),(w,h),bar_color,-1)
    cv2.addWeighted(overlay2,0.75,frame,0.25,0,frame)
    cv2.putText(frame,status_text,(10,h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,status_color,1,cv2.LINE_AA)
    
    # # Day label
    # cv2.putText(frame, "DAY 1-2: TOPOLOGY", (w - 175, h - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
=======
    status_text  = "FACE DETECTED" if face_detected else "NO FACE — INTERLOCK"
    status_color = (0, 220, 120)   if face_detected else (0, 60, 220)
    overlay2 = frame.copy()
    bar_color = (20, 60, 20) if face_detected else (20, 20, 120)
    cv2.rectangle(overlay2, (0, h - 32), (w, h), bar_color, -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, status_text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)

    # Day label
    cv2.putText(frame, "DAY 1-2: TOPOLOGY", (w - 175, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f


# ──────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────

def main():
    print("="*55)
    print(" AI Driver Monitoring — Day 1-2")
    print(" Webcam thread + MediaPipe face mesh + HUD")
    print("="*55)
    print(" Controls:")
    print("   Q  — quit")
    print("   M  — toggle full mesh overlay")
    print("="*55)

    capture = CaptureThread(src=0)
    capture.start()

    show_mesh   = False
    fps_timer   = time.time()
    fps_counter = 0
    fps_display = 0.0
<<<<<<< HEAD
    eye_closed_since=None
    drowsy_alert=False
=======
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f

    while True:
        frame = capture.get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        # ── MediaPipe inference ──────────────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        face_detected = result.multi_face_landmarks is not None
<<<<<<< HEAD
        avg_ear = None
        eye_state = "NO FACE"
        closed_duration=0.0
=======
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f

        if face_detected:
            lms = result.multi_face_landmarks[0].landmark

            # Optional: full mesh overlay
            if show_mesh:
                mp_drawing.draw_landmarks(
                    image            = frame,
                    landmark_list    = result.multi_face_landmarks[0],
                    connections      = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_styles
                        .get_default_face_mesh_tesselation_style()
                )

            # ── Eye landmarks (EAR points — used from Day 3) ──
            le_pts = draw_landmark_group(
                frame, lms, LEFT_EYE,  w, h,
                color=(0, 220, 120), label="L-EYE")
            re_pts = draw_landmark_group(
                frame, lms, RIGHT_EYE, w, h,
                color=(0, 220, 120), label="R-EYE")
            draw_bbox(frame, le_pts, (0, 200, 100))
            draw_bbox(frame, re_pts, (0, 200, 100))

<<<<<<< HEAD
            left_ear=eye_aspect_ratio(lms,LEFT_EYE,w,h)
            right_ear=eye_aspect_ratio(lms,RIGHT_EYE,w,h)
            avg_ear=(left_ear+right_ear)/2.0

            if avg_ear < EAR_THRESHOLD:
                if eye_closed_since is None:
                    eye_closed_since = time.time()
                closed_duration =time.time() - eye_closed_since
                drowsy_alert = closed_duration >=DROWSY_SECONDS or avg_ear < MICROSLEEP_THRESHOLD
                eye_state="DROWSY" if drowsy_alert else "EYES CLOSED"
            else:
                eye_closed_since=None
                drowsy_alert=False
                eye_state="EYES OPEN"

=======
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f
            # ── Mouth landmarks (MAR points — used from Day 4) ──
            m_pts = draw_landmark_group(
                frame, lms, LIPS, w, h,
                color=(0, 165, 255), label="MOUTH")
            draw_bbox(frame, m_pts, (0, 140, 220))

            # ── Nose tip (PnP reference — used from Day 5) ──
            nx, ny = lm_to_px(lms[NOSE_TIP], w, h)
            cv2.circle(frame, (nx, ny), 3, (200, 80, 255), -1)
            cv2.putText(frame, "NOSE", (nx + 5, ny - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 80, 255),
                        1, cv2.LINE_AA)

            # ── Info panel (right side) ──────────────────────
            info_lines = [
<<<<<<< HEAD
                                
                ("468 landmarks: OK",  (0, 220, 120)),
                (f"EAR: {avg_ear:.2f}", (0, 220, 120) if avg_ear >= EAR_THRESHOLD else (0, 165, 255)),
                (f"Eyes: {eye_state}",  (0, 220, 120) if eye_state == "EYES OPEN" else (0, 165, 255)),
                (f"Closed: {closed_duration:.1f}s", (150, 150, 150)),
                ("DROWSY ALERT" if drowsy_alert else "Status: normal",
                 (0, 60, 220) if drowsy_alert else (0, 220, 120))
=======
                ("DAY 1-2 COMPLETE",   (200, 200, 200)),
                ("468 landmarks: OK",  (0, 220, 120)),
                ("Eye indices: ready", (0, 220, 120)),
                ("Mouth indices: ready",(0, 165, 255)),
                ("Nose tip: ready",    (200, 80, 255)),
                ("Next: EAR (Day 3)",  (150, 150, 150)),
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f
            ]
            panel_x = w - 200
            panel_y = 50
            overlay3 = frame.copy()
            cv2.rectangle(overlay3,
                          (panel_x - 8, panel_y - 18),
                          (w - 4, panel_y + len(info_lines) * 20 + 4),
                          (15, 15, 30), -1)
            cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
            for i, (txt, col) in enumerate(info_lines):
                cv2.putText(frame, txt,
                            (panel_x, panel_y + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, col, 1, cv2.LINE_AA)
<<<<<<< HEAD
            if drowsy_alert:
                cv2.putText(frame,"DROWSY ALERT",
                            (w//2-120,75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,(0,60,255),2,cv2.LINE_AA)

        # ── HUD ─────────────────────────────────────────
        if not face_detected:
            eye_closed_since=None
            drowsy_alert=False
        
        draw_hud(frame, fps_display, face_detected, w, h,
                 "DROWSY" if drowsy_alert else eye_state)

        cv2.imshow("AI Driver Monitoring ", frame)
=======

        # ── HUD ─────────────────────────────────────────
        draw_hud(frame, fps_display, face_detected, w, h)

        cv2.imshow("AI Driver Monitoring — Day 1-2", frame)
>>>>>>> 24d7b1db232173575070297d019def450a2acc7f

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mesh = not show_mesh
            print(f"Mesh overlay: {'ON' if show_mesh else 'OFF'}")

    capture.stop()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Session ended.")


if __name__ == "__main__":
    main()
