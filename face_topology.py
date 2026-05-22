"""
=============================================================
 AI Driver Monitoring System
 DAY 4: EAR + MAR + No-Person Interlock
=============================================================
 Timeline reference: Day 4 - Yawn Detection and Interlock
 - Async webcam thread (Thread 1)
 - 480p downscale → frame queue
 - MediaPipe 468-point face mesh
 - Map landmark indices for eyes and lips
 - EAR drowsiness detection
 - MAR yawning detection
 - No-person interlock state
 - HUD overlay: FPS, landmark dots, EAR, MAR, state ID
=============================================================
DAY4
- MAR calculation from mouth landmarks
- Clock based yawning alert when MAR stays high
- Interlock state when no face is detected
=============================================================
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

# EAR THRESHOLDS
EAR_THRESHOLD= 0.18
MICROSLEEP_THRESHOLD=0.10
DROWSY_SECONDS=1.5

MAR_THRESHOLD=0.60
YAWN_SECONDS=3.0
ABSENT_SECONDS=1.0

STATE_NORMAL=0
STATE_YAWNING=1
STATE_DROWSY=2
STATE_ABSENT=4

LIGHT_GREEN=(120,255,120)
LIGHT_GREEN_BOX=(70,255,140)
NORMAL_GREEN=(0,220,120)
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
def mouth_aspect_ratio(landmarks,w,h):
    """"Calculate MAR using 4 Mouth Landmarks"""
    top_lip=lm_to_px(landmarks[MOUTH[0]],w,h)
    bottom_lip=lm_to_px(landmarks[MOUTH[1]],w,h)
    left_point=lm_to_px(landmarks[MOUTH[2]],w,h)
    right_point=lm_to_px(landmarks[MOUTH[3]],w,h)

    vertical=euclidean_distance(top_lip,bottom_lip)
    horizontal=euclidean_distance(left_point,right_point)

    if horizontal==0:
        return 0.0
    
    return vertical/horizontal


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

def draw_bbox(frame,pts,color,padding=4):
    if not pts:
        return
    xs=[p[0] for p in pts]
    ys=[p[1] for p in pts]
    cv2.rectangle(
            frame,
            (min(xs)-padding,min(ys)-padding),
            (max(xs)+padding,max(ys)+padding),
            color,
            1,
    )

def draw_top_hud(frame,fps,w):
    overlay=frame.copy()
    cv2.rectangle(overlay,(0,0),(w,38) ,(15,15,30),-1)
    cv2.addWeighted(overlay,0.7,frame,0.3,0,frame)

    cv2.putText(
        frame,
        "AI DRIVER MONITERING SYSTEM",
        (10,24),cv2.FONT_HERSHEY_SIMPLEX,0.58,
        (255,255,255),1,cv2.LINE_AA
    )

    fps_color=(0,220,120) if fps>=15 else (0,165,255)
    cv2.putText(
            frame,f"FPS: {fps:.1f}",
            (w-110,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,
            fps_color,1,cv2.LINE_AA
    )

def draw_bottom_status(frame, status_text, status_color, w, h, state_id):
    overlay = frame.copy()
    if state_id in (STATE_DROWSY, STATE_ABSENT):
        bar_color = (20, 20, 140)
    elif state_id == STATE_YAWNING:
        bar_color = (20, 80, 130)
    else:
        bar_color = (20, 70, 20)

    cv2.rectangle(overlay, (0, h - 34), (w, h), bar_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(
        frame, status_text, (10, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        status_color, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame, f"STATE ID: {state_id}",
        (w - 130, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (180, 180, 180), 1, cv2.LINE_AA
    )

def draw_info_panel(frame, lines, w):
    panel_x = w - 220
    panel_y = 55
    line_h = 20
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x - 8, panel_y - 18),
        (w - 4, panel_y + len(lines) * line_h + 4),
        (15, 15, 30),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, (txt, col) in enumerate(lines):
        cv2.putText(
            frame, txt, (panel_x, panel_y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            col, 1, cv2.LINE_AA
        )

def resolve_state(face_detected,absent_duration,drowsy_alert,yawn_alert):
    if not face_detected and absent_duration >=ABSENT_SECONDS:
        return STATE_ABSENT,"DRIVER ABSENT - INTERLOCK",(0,60,255)
    if drowsy_alert:
        return STATE_DROWSY,"DROWSY ALERT",(0,60,255)
    if yawn_alert:
        return STATE_YAWNING,"YAWNING ALERT",(0,165,255)
    if not face_detected:
        return STATE_ABSENT,"NO FACE DETECTED",(0,165,255)
    return STATE_NORMAL,"ALERT / NORMAL",NORMAL_GREEN


# ──────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────

def main():
    print("="*60)
    print(" AI Driver Monitoring System")
    print(" EAR + MAR + NO-PERSON INTERLOCK")
    print("="*60)
    print(" Controls:")
    print("   Q  — quit")
    print("   M  — toggle full mesh overlay")
    print("="*60)

    capture = CaptureThread(src=0)
    capture.start()

    show_mesh   = False
    fps_timer   = time.time()
    fps_counter = 0
    fps_display = 0.0

    eye_closed_since=None
    mouth_open_since=None
    no_face_since=None

    drowsy_alert=False
    yawn_alert=False

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

        avg_ear = None
        mar=None
        eye_state = "NO FACE"
        mouth_state="NO FACE"
        closed_duration=0.0
        yawn_duration=0.0
        absent_duration=0.0

        if face_detected:
            no_face_since=None
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
                color=LIGHT_GREEN, label="L-EYE")
            re_pts = draw_landmark_group(
                frame, lms, RIGHT_EYE, w, h,
                color=LIGHT_GREEN, label="R-EYE")
            draw_bbox(frame, le_pts, LIGHT_GREEN_BOX)
            draw_bbox(frame, re_pts, LIGHT_GREEN_BOX)

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

            # ── Mouth landmarks (MAR points — used from Day 4) ──

            lip_pts=draw_landmark_group(frame,lms,LIPS,w,h,(0,165,255),"MOUTH")
            key_mouth_pts=draw_landmark_group(frame,lms,MOUTH,w,h,(0,255,255))
            draw_bbox(frame,lip_pts,(0,140,200))
            mar=mouth_aspect_ratio(lms,w,h)

            if mar > MAR_THRESHOLD:
                if mouth_open_since is None:
                    mouth_open_since =time.time()
                yawn_duration=time.time() - mouth_open_since
                yawn_alert = yawn_duration>=YAWN_SECONDS
                mouth_state="YAWNING" if yawn_alert else "MOUTH OPEN"
            else:
                mouth_open_since=None
                yawn_alert=False
                mouth_state="NORMAL"

            if len(key_mouth_pts)==4:
                cv2.line(frame,key_mouth_pts[0],key_mouth_pts[1],(0,255,255),1)
                cv2.line(frame,key_mouth_pts[2],key_mouth_pts[3],(0,255,255),1)

            # ── Nose tip (PnP reference — used from Day 5) ──
            nx, ny = lm_to_px(lms[NOSE_TIP], w, h)
            cv2.circle(frame, (nx, ny), 3, (200, 80, 255), -1)
            cv2.putText(frame, "NOSE", (nx + 5, ny - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 80, 255),
                        1, cv2.LINE_AA)
            
        else:
            eye_closed_since=None
            mouth_open_since=None
            drowsy_alert=False
            yawn_alert=False

            if no_face_since is None:
                no_face_since=time.time()
            absent_duration=time.time() - no_face_since

        state_id, status_text, status_color = resolve_state(
            face_detected, absent_duration, drowsy_alert, yawn_alert
        )

        lines = [
            ("EAR + MAR + INTERLOCK", (200, 200, 200)),
            
            ("Face: detected" if face_detected else "Face: not detected",
             NORMAL_GREEN if face_detected else (0, 165, 255)),
            
            (f"EAR: {avg_ear:.2f}" if avg_ear is not None else "EAR: --",
             NORMAL_GREEN if avg_ear is not None and avg_ear >= EAR_THRESHOLD else (0, 165, 255)),
           
            (f"MAR: {mar:.2f}" if mar is not None else "MAR: --",
             NORMAL_GREEN if mar is not None and mar <= MAR_THRESHOLD else (0, 165, 255)),
           
            (f"Eyes: {eye_state}", NORMAL_GREEN if eye_state == "EYES OPEN" else (0, 165, 255)),
           
            (f"Mouth: {mouth_state}", NORMAL_GREEN if mouth_state == "NORMAL" else (0, 165, 255)),
           
            (f"Yawn timer: {yawn_duration:.1f}s", (150, 150, 150)),
           
            (f"Absent timer: {absent_duration:.1f}s", (150, 150, 150)),
           
            (f"State ID: {state_id}", status_color),
        ]

        draw_top_hud(frame, fps_display, w)
        draw_info_panel(frame, lines, w)
        draw_bottom_status(frame, status_text, status_color, w, h, state_id)

        if yawn_alert:
            cv2.putText(
                frame, "YAWNING DETECTED",
                (w // 2 - 145, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 165, 255), 2, cv2.LINE_AA
            )
        elif drowsy_alert:
            cv2.putText(
                frame, "DROWSY ALERT",
                (w // 2 - 120, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 60, 255), 2, cv2.LINE_AA
            )
        elif state_id == STATE_ABSENT and absent_duration >= ABSENT_SECONDS:
            cv2.putText(
                frame, "SYSTEM INTERLOCK",
                (w // 2 - 150, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 60, 255), 2, cv2.LINE_AA
            )

        cv2.imshow("AI Driver Monitoring - Day 4 MAR", frame)

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
