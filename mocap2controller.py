"""
Motion Controller
-----------------
Maps body landmarks (via MediaPipe HolisticLandmarker) to a virtual Xbox controller (via vgamepad).
Uses OpenCV for webcam capture and PySimpleGUI for live config.

Controls:
  - Waist Y > upper_bound  → A button
  - Waist Y < lower_bound  → B button
  - Left  hand → Left stick axes
  - Right hand → Right stick axes
  - Right hand distance > right-stick radius → Right stick click (sprint, toggleable)
  - Left  fist held 1 s   → Radial menu (hand drives selection, open hand confirms)
  - Right fist             → RT (firing mode) or LT (melee mode)

Requirements:
    pip install opencv-python mediapipe vgamepad PySimpleGUI numpy requests
"""

import math
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import PySimpleGUI as sg

try:
    import vgamepad as vg
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    print("[WARN] vgamepad not found – running in preview-only mode.")

# ─────────────────────────────────────────────
#  Model download
# ─────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "holistic_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading holistic_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved to {MODEL_PATH}")

# ─────────────────────────────────────────────
#  Camera enumeration
# ─────────────────────────────────────────────
def enumerate_cameras(max_index=8):
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
            cap.release()
    return found if found else [0]

# ─────────────────────────────────────────────
#  Camera brightness (set once at open time)
# ─────────────────────────────────────────────
CAM_BRIGHTNESS = 100   # 0–255 typical range; adjust to taste

# ─────────────────────────────────────────────
#  Radial menu config
# ─────────────────────────────────────────────
RADIAL_LABELS = ["Firing Mode", "Swap Weapon (X)", "Melee Mode", "Reload (Y)"]
RADIAL_COLORS = [
    (0,   200, 255),   # right  – firing  (cyan)
    (0,   255, 100),   # up     – swap    (green)
    (255, 160,   0),   # left   – melee   (orange)
    (180,   0, 255),   # down   – reload  (purple)
]
RADIAL_ANGLES = [0, -90, 180, 90]   # right, up, left, down

RADIAL_HOLD_SECS = 1.0
RADIAL_DEAD_ZONE = 0.45

# ─────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────
DEFAULT = dict(
    waist_upper=0.35,
    waist_lower=0.65,
    left_cx=0.25,
    left_cy=0.50,
    left_r=0.15,
    right_cx=0.75,
    right_cy=0.50,
    right_r=0.15,
    sprint_enabled=True,
)

# ─────────────────────────────────────────────
#  Shared state
# ─────────────────────────────────────────────
state_lock = threading.Lock()
state = dict(DEFAULT)
state.update(dict(
    frame=None,
    running=True,
    waist_norm=None,
    lh_norm=None,
    rh_norm=None,
    btn_a=False,
    btn_b=False,
    rs_click=False,
    left_axis=(0.0, 0.0),
    right_axis=(0.0, 0.0),
    lh_fist=False,
    rh_fist=False,
    radial_open=False,
    radial_hovered=-1,
    combat_mode="firing",
    cam_index=1,
    cam_requested=None,
    cam_actual=1,
))

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def hand_to_axis(hand_xy, center_xy, radius):
    dx = hand_xy[0] - center_xy[0]
    dy = hand_xy[1] - center_xy[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, 0.0
    scale = min(dist / radius, 1.0)
    angle = math.atan2(dy, dx)
    return math.cos(angle) * scale, -math.sin(angle) * scale

# ─────────────────────────────────────────────
#  Fist detection
# ─────────────────────────────────────────────
_TIPS = [8, 12, 16, 20]
_MIDS = [6, 10, 14, 18]

def is_fist(landmarks, threshold=0.8):
    if not landmarks or len(landmarks) < 21:
        return False
    wrist = landmarks[0]
    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)
    curled = sum(
        1 for tip_idx, mid_idx in zip(_TIPS, _MIDS)
        if dist(landmarks[mid_idx], wrist) > 0
        and dist(landmarks[tip_idx], wrist) < dist(landmarks[mid_idx], wrist) * threshold
    )
    return curled >= 3

# ─────────────────────────────────────────────
#  Radial menu: angle → sector index
# ─────────────────────────────────────────────
def angle_to_sector(dx, dy):
    """
    Maps a hand delta (dx, dy) in screen coords (+Y down) to a sector 0-3:
      0 = right  (firing mode)
      1 = up     (swap weapon)
      2 = left   (melee mode)
      3 = down   (reload)

    +45° offset centres sector boundaries at the diagonals so that
    cardinal directions (up/down/left/right) map cleanly to sector midpoints.
    """
    angle_deg = math.degrees(math.atan2(dy, dx))
    adjusted  = (angle_deg + 45) % 360
    bin_      = int(adjusted // 90) % 4
    # bin 0 (0–90°)   → RIGHT → sector 0
    # bin 1 (90–180°) → DOWN  → sector 3
    # bin 2 (180–270°)→ LEFT  → sector 2
    # bin 3 (270–360°)→ UP    → sector 1
    REMAP = [0, 3, 2, 1]
    return REMAP[bin_]

# ─────────────────────────────────────────────
#  Radial menu drawing
# ─────────────────────────────────────────────
def draw_radial_menu(frame, center_px, radius_px, hovered_sector, combat_mode):
    cx, cy = center_px
    overlay = frame.copy()

    for i in range(4):
        start_angle = i * 90 - 45
        end_angle   = start_angle + 90
        color = RADIAL_COLORS[i]

        pts = [(cx, cy)]
        for a in range(int(start_angle), int(end_angle) + 1, 2):
            rad = math.radians(a)
            pts.append((
                int(cx + radius_px * math.cos(rad)),
                int(cy + radius_px * math.sin(rad)),
            ))
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)

    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i in range(4):
        angle_rad = math.radians(i * 90 - 45)
        ex = int(cx + radius_px * math.cos(angle_rad))
        ey = int(cy + radius_px * math.sin(angle_rad))
        cv2.line(frame, (cx, cy), (ex, ey), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), radius_px, (255, 255, 255), 2, cv2.LINE_AA)
    dead_r = int(radius_px * RADIAL_DEAD_ZONE)
    cv2.circle(frame, (cx, cy), dead_r, (200, 200, 200), 1, cv2.LINE_AA)

    label_r = int(radius_px * 0.68)
    for i, (label, angle_deg) in enumerate(zip(RADIAL_LABELS, RADIAL_ANGLES)):
        rad = math.radians(angle_deg)
        lx = int(cx + label_r * math.cos(rad))
        ly = int(cy + label_r * math.sin(rad))
        color = (255, 255, 255) if i == hovered_sector else (200, 200, 200)
        fw = 2 if i == hovered_sector else 1
        cv2.putText(frame, label, (lx - 1, ly + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), fw + 1, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, fw, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  Mode display (bottom-left HUD)
# ─────────────────────────────────────────────
def draw_mode_label(frame, combat_mode, h, w):
    label = f"MODE: {combat_mode.upper()}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thick = 3
    x = 16
    y = h - 20
    cv2.putText(frame, label, (x, y), font, scale, (0, 0, 200), thick + 4, cv2.LINE_AA)
    cv2.putText(frame, label, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  Controller wrapper
# ─────────────────────────────────────────────
class VirtualController:
    def __init__(self):
        self.gamepad = None
        if CONTROLLER_AVAILABLE:
            try:
                self.gamepad = vg.VX360Gamepad()
                print("[INFO] Virtual Xbox360 controller created via ViGEmBus.")
            except Exception as e:
                print(f"[WARN] Could not create virtual controller: {e}")

    def set_button(self, button_const, pressed: bool):
        if not self.gamepad or button_const is None:
            return
        try:
            if pressed:
                self.gamepad.press_button(button=button_const)
            else:
                self.gamepad.release_button(button=button_const)
        except Exception:
            pass

    def set_axes(self, lx, ly, rx, ry):
        if not self.gamepad:
            return
        try:
            self.gamepad.left_joystick_float(x_value_float=lx, y_value_float=ly)
            self.gamepad.right_joystick_float(x_value_float=rx, y_value_float=ry)
        except Exception:
            pass

    def set_trigger(self, left_val, right_val):
        if not self.gamepad:
            return
        try:
            self.gamepad.left_trigger_float(value_float=left_val)
            self.gamepad.right_trigger_float(value_float=right_val)
        except Exception:
            pass

    def flush(self):
        if self.gamepad:
            try:
                self.gamepad.update()
            except Exception:
                pass

    def reset(self):
        if self.gamepad:
            try:
                self.gamepad.reset()
                self.gamepad.update()
            except Exception:
                pass

# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),
    (0,9),(0,13),(0,17),(5,9),(9,13),(13,17),
]

def draw_landmarks(frame, landmarks, connections, h, w,
                   dot_color=(0,255,0), line_color=(0,200,0), dot_r=4, thickness=2):
    if not landmarks:
        return
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, thickness)
    for pt in pts:
        cv2.circle(frame, pt, dot_r, dot_color, -1)

# ─────────────────────────────────────────────
#  Camera open (with brightness)
# ─────────────────────────────────────────────
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, CAM_BRIGHTNESS)
    return cap

# ─────────────────────────────────────────────
#  Capture thread
# ─────────────────────────────────────────────
def capture_thread():
    ensure_model()

    BaseOptions               = mp.tasks.BaseOptions
    HolisticLandmarker        = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode         = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    with state_lock:
        current_index = state["cam_index"]

    cap = open_camera(current_index)
    with state_lock:
        state["cam_actual"] = current_index

    ctrl = VirtualController()

    BTN_A  = vg.XUSB_BUTTON.XUSB_GAMEPAD_A            if CONTROLLER_AVAILABLE else None
    BTN_B  = vg.XUSB_BUTTON.XUSB_GAMEPAD_B            if CONTROLLER_AVAILABLE else None
    BTN_RS = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB  if CONTROLLER_AVAILABLE else None
    BTN_X  = vg.XUSB_BUTTON.XUSB_GAMEPAD_X            if CONTROLLER_AVAILABLE else None
    BTN_Y  = vg.XUSB_BUTTON.XUSB_GAMEPAD_Y            if CONTROLLER_AVAILABLE else None

    frame_ts = 0
    lh_fist_since      = None
    radial_open        = False
    radial_action_done = False

    with HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            with state_lock:
                if not state["running"]:
                    break
                requested = state["cam_requested"]
                if requested is not None and requested != current_index:
                    state["cam_requested"] = None
                else:
                    requested = None

            if requested is not None:
                print(f"[INFO] Switching camera {current_index} → {requested}")
                cap.release()
                cap = open_camera(requested)
                current_index = requested
                with state_lock:
                    state["cam_actual"] = current_index
                    state["frame"] = np.zeros((480, 640, 3), dtype=np.uint8)

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            result = landmarker.detect_for_video(mp_image, frame_ts)

            with state_lock:
                wu          = state["waist_upper"]
                wl          = state["waist_lower"]
                lcx         = state["left_cx"];  lcy = state["left_cy"];  lr = state["left_r"]
                rcx         = state["right_cx"]; rcy = state["right_cy"]; rr = state["right_r"]
                sprint_on   = state["sprint_enabled"]
                combat_mode = state["combat_mode"]

            # ── Pose ──────────────────────────────────────────────────
            waist_norm = None
            pose_lms = result.pose_landmarks
            if pose_lms and len(pose_lms) > 24:
                waist_norm = (pose_lms[23].y + pose_lms[24].y) / 2.0

            # ── Hands (frame is mirrored) ──────────────────────────────
            rh_lms = result.left_hand_landmarks
            lh_lms = result.right_hand_landmarks

            rh_norm = (rh_lms[0].x, rh_lms[0].y) if rh_lms else None
            lh_norm = (lh_lms[0].x, lh_lms[0].y) if lh_lms else None

            rh_fist = is_fist(rh_lms)
            lh_fist = is_fist(lh_lms)

            now = time.monotonic()

            # ── Left-fist hold → radial menu ──────────────────────────
            if lh_fist:
                if lh_fist_since is None:
                    lh_fist_since = now
                held_secs = now - lh_fist_since
                if held_secs >= RADIAL_HOLD_SECS and not radial_open:
                    radial_open        = True
                    radial_action_done = False
            else:
                if radial_open and not radial_action_done:
                    if lh_norm:
                        dx = lh_norm[0] - lcx
                        dy = lh_norm[1] - lcy
                        dist_frac = math.hypot(dx, dy) / lr if lr > 0 else 0
                        if dist_frac >= RADIAL_DEAD_ZONE:
                            sector = angle_to_sector(dx, dy)
                            with state_lock:
                                if sector == 0:
                                    state["combat_mode"] = "firing"
                                elif sector == 2:
                                    state["combat_mode"] = "melee"
                            if sector == 1:
                                ctrl.set_button(BTN_X, True);  ctrl.flush()
                                time.sleep(0.08)
                                ctrl.set_button(BTN_X, False); ctrl.flush()
                            elif sector == 3:
                                ctrl.set_button(BTN_Y, True);  ctrl.flush()
                                time.sleep(0.08)
                                ctrl.set_button(BTN_Y, False); ctrl.flush()
                    radial_action_done = True

                radial_open   = False
                lh_fist_since = None

            # Hovered sector for visual feedback
            hovered_sector = -1
            if radial_open and lh_norm:
                dx = lh_norm[0] - lcx
                dy = lh_norm[1] - lcy
                dist_frac = math.hypot(dx, dy) / lr if lr > 0 else 0
                if dist_frac >= RADIAL_DEAD_ZONE:
                    hovered_sector = angle_to_sector(dx, dy)

            # ── Standard controls ──────────────────────────────────────
            btn_a = bool(waist_norm is not None and waist_norm < wu)
            btn_b = bool(waist_norm is not None and waist_norm > wl)

            if radial_open:
                left_axis = (0.0, 0.0)
            else:
                left_axis = hand_to_axis(lh_norm, (lcx, lcy), lr) if lh_norm else (0.0, 0.0)

            right_axis = hand_to_axis(rh_norm, (rcx, rcy), rr) if rh_norm else (0.0, 0.0)

            rs_click = False
            if sprint_on and rh_norm:
                d = math.hypot(rh_norm[0] - rcx, rh_norm[1] - rcy)
                rs_click = d > rr

            with state_lock:
                combat_mode = state["combat_mode"]

            if combat_mode == "firing":
                lt_val = 0.0
                rt_val = 1.0 if rh_fist else 0.0
            else:
                lt_val = 1.0 if rh_fist else 0.0
                rt_val = 0.0

            ctrl.set_button(BTN_A,  btn_a)
            ctrl.set_button(BTN_B,  btn_b)
            ctrl.set_button(BTN_RS, rs_click)
            ctrl.set_trigger(lt_val, rt_val)
            ctrl.set_axes(left_axis[0], left_axis[1], right_axis[0], right_axis[1])
            ctrl.flush()

            # ── Draw overlays ──────────────────────────────────────────
            if pose_lms:
                draw_landmarks(frame, pose_lms, POSE_CONNECTIONS, h, w,
                               dot_color=(66,117,245), line_color=(230,66,245))
            if lh_lms:
                draw_landmarks(frame, lh_lms, HAND_CONNECTIONS, h, w,
                               dot_color=(0,200,255), line_color=(0,150,200))
            if rh_lms:
                draw_landmarks(frame, rh_lms, HAND_CONNECTIONS, h, w,
                               dot_color=(255,200,0), line_color=(200,150,0))

            cv2.line(frame, (0, int(wu*h)), (w, int(wu*h)), (0,255,0), 2)
            cv2.putText(frame, "A (raise above)", (5, int(wu*h)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.line(frame, (0, int(wl*h)), (w, int(wl*h)), (0,120,255), 2)
            cv2.putText(frame, "B (crouch below)", (5, int(wl*h)+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,120,255), 2)
            if waist_norm is not None:
                wy = int(waist_norm * h)
                cv2.line(frame, (0, wy), (w, wy), (255,255,0), 2)
                cv2.putText(frame, "Waist", (5, wy-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

            lc_px = (int(lcx * w), int(lcy * h))
            rc_px = (int(rcx * w), int(rcy * h))
            if not radial_open:
                cv2.circle(frame, lc_px, int(lr*w), (0,200,255), 2)
                cv2.putText(frame, "L", (lc_px[0]-6, lc_px[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)
            cv2.circle(frame, rc_px, int(rr*w),
                       (0,60,255) if sprint_on else (160,160,160), 2)
            cv2.putText(frame, "R", (rc_px[0]-6, rc_px[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,60,255), 2)

            if radial_open:
                draw_radial_menu(frame, lc_px, int(lr*w), hovered_sector, combat_mode)
            elif lh_fist and lh_fist_since is not None:
                held     = now - lh_fist_since
                progress = min(held / RADIAL_HOLD_SECS, 1.0)
                radius_px = int(lr * w)
                cv2.ellipse(frame, lc_px, (radius_px, radius_px),
                            -90, 0, int(360 * progress), (0,220,255), 3, cv2.LINE_AA)
                cv2.putText(frame, "Hold...", (lc_px[0]-28, lc_px[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,255), 1)

            if lh_norm:
                lh_px = (int(lh_norm[0]*w), int(lh_norm[1]*h))
                lh_col = (0,0,200) if lh_fist else (0,200,255)
                cv2.circle(frame, lh_px, 12, lh_col, -1)
                cv2.putText(frame, "FIST" if lh_fist else "open",
                            (lh_px[0]+14, lh_px[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0,80,255) if lh_fist else (0,200,255), 1)
                if not radial_open:
                    cv2.line(frame, lh_px, lc_px, (0,200,255), 2)
            if rh_norm:
                rh_px = (int(rh_norm[0]*w), int(rh_norm[1]*h))
                rh_col = (0,0,200) if rh_fist else ((0,0,180) if rs_click else (0,60,255))
                cv2.circle(frame, rh_px, 12, rh_col, -1)
                cv2.putText(frame, "FIST" if rh_fist else "open",
                            (rh_px[0]+14, rh_px[1]+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0,0,255) if rh_fist else (0,60,255), 1)
                cv2.line(frame, rh_px, rc_px, (0,60,255), 2)

            badge = f"CAM {current_index}"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (w-bw-12, 4), (w-4, bh+10), (30,30,30), -1)
            cv2.putText(frame, badge, (w-bw-8, bh+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,180), 2)

            draw_mode_label(frame, combat_mode, h, w)

            hud = [
                f"A: {'ON' if btn_a else 'off'}   B: {'ON' if btn_b else 'off'}"
                f"   Sprint RS: {'ON' if rs_click else 'off'}",
                f"L ({left_axis[0]:+.2f}, {left_axis[1]:+.2f})"
                f"   R ({right_axis[0]:+.2f}, {right_axis[1]:+.2f})",
                f"LH: {'MENU' if radial_open else ('FIST' if lh_fist else 'open')}"
                f"   RH: {'FIST→' + ('RT' if combat_mode=='firing' else 'LT') if rh_fist else 'open'}",
            ]
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (8, 22+i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

            disp_w = min(w, 640)
            disp_h = int(h * disp_w / w)
            display = cv2.resize(frame, (disp_w, disp_h))

            with state_lock:
                state["frame"]          = display
                state["waist_norm"]     = waist_norm
                state["lh_norm"]        = lh_norm
                state["rh_norm"]        = rh_norm
                state["btn_a"]          = btn_a
                state["btn_b"]          = btn_b
                state["rs_click"]       = rs_click
                state["left_axis"]      = left_axis
                state["right_axis"]     = right_axis
                state["lh_fist"]        = lh_fist
                state["rh_fist"]        = rh_fist
                state["radial_open"]    = radial_open
                state["radial_hovered"] = hovered_sector

    ctrl.reset()
    cap.release()

# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
def slider(key, label, lo, hi, default, resolution=0.01):
    return [
        sg.Text(label, size=(24,1)),
        sg.Slider(range=(lo,hi), default_value=default, resolution=resolution,
                  orientation="h", size=(28,15), key=key, enable_events=True),
    ]

def build_layout(cam_list, initial_cam):
    frame_col = [[sg.Image(key="-FRAME-", size=(640,480))]]
    cam_labels    = [f"Camera {i}" for i in cam_list]
    initial_label = f"Camera {initial_cam}" if initial_cam in cam_list else cam_labels[0]

    controls_col = [
        [sg.Frame("Camera", [[
            sg.Text("Source:", size=(8,1)),
            sg.Combo(cam_labels, default_value=initial_label,
                     key="-CAM_SELECT-", size=(14,1), readonly=True, enable_events=True),
            sg.Button("Switch", key="-CAM_SWITCH-", size=(7,1)),
            sg.Text("", key="-CAM_STATUS-", size=(14,1), text_color="yellow"),
        ]])],
        [sg.Frame("Waist Bounds", [
            slider("-WAIST_UP-",  "Upper (A: raise above)", 0.0, 1.0, DEFAULT["waist_upper"]),
            slider("-WAIST_LOW-", "Lower (B: crouch below)", 0.0, 1.0, DEFAULT["waist_lower"]),
        ])],
        [sg.Frame("Left Stick Circle", [
            slider("-LCX-", "Center X", 0.0, 1.0, DEFAULT["left_cx"]),
            slider("-LCY-", "Center Y", 0.0, 1.0, DEFAULT["left_cy"]),
            slider("-LR-",  "Radius",   0.0, 0.5, DEFAULT["left_r"]),
        ])],
        [sg.Frame("Right Stick Circle", [
            slider("-RCX-", "Center X", 0.0, 1.0, DEFAULT["right_cx"]),
            slider("-RCY-", "Center Y", 0.0, 1.0, DEFAULT["right_cy"]),
            slider("-RR-",  "Radius",   0.0, 0.5, DEFAULT["right_r"]),
        ])],
        [sg.Frame("Sprint (RS Click)", [
            [sg.Checkbox("Enable sprint when right hand exceeds radius",
                         default=DEFAULT["sprint_enabled"],
                         key="-SPRINT-", enable_events=True)],
        ])],
        [sg.HSeparator()],
        [sg.Frame("Live Feedback", [
            [sg.Text("", key="-FB_WAIST-", size=(44,1))],
            [sg.Text("", key="-FB_BTNS-",  size=(44,1))],
            [sg.Text("", key="-FB_LAXIS-", size=(44,1))],
            [sg.Text("", key="-FB_RAXIS-", size=(44,1))],
            [sg.Text("", key="-FB_FIST-",  size=(44,1))],
            [sg.Text("", key="-FB_MODE-",  size=(44,1), font=("Helvetica", 11, "bold"))],
            [sg.Text("", key="-FB_MENU-",  size=(44,1))],
        ])],
        [sg.Button("Reset Defaults", key="-RESET-"),
         sg.Button("Quit",           key="-QUIT-")],
    ]

    return [[
        sg.Column(frame_col,    vertical_alignment="top"),
        sg.VSeparator(),
        sg.Column(controls_col, vertical_alignment="top"),
    ]]

def apply_defaults(window):
    for key, val in [
        ("-WAIST_UP-", DEFAULT["waist_upper"]), ("-WAIST_LOW-", DEFAULT["waist_lower"]),
        ("-LCX-", DEFAULT["left_cx"]), ("-LCY-", DEFAULT["left_cy"]), ("-LR-", DEFAULT["left_r"]),
        ("-RCX-", DEFAULT["right_cx"]), ("-RCY-", DEFAULT["right_cy"]), ("-RR-", DEFAULT["right_r"]),
    ]:
        window[key].update(val)
    window["-SPRINT-"].update(DEFAULT["sprint_enabled"])

SLIDER_MAP = {
    "-WAIST_UP-": "waist_upper", "-WAIST_LOW-": "waist_lower",
    "-LCX-": "left_cx", "-LCY-": "left_cy", "-LR-": "left_r",
    "-RCX-": "right_cx", "-RCY-": "right_cy", "-RR-": "right_r",
}

def run_gui(cam_list):
    sg.theme("DarkBlue3")
    with state_lock:
        initial_cam = state["cam_index"]

    window = sg.Window("Motion Controller", build_layout(cam_list, initial_cam), finalize=True)
    switching     = False
    switch_target = None

    while True:
        event, values = window.read(timeout=30)
        if event in (sg.WIN_CLOSED, "-QUIT-"):
            with state_lock:
                state["running"] = False
            break

        if event == "-RESET-":
            apply_defaults(window)

        if event == "-CAM_SWITCH-":
            label = values["-CAM_SELECT-"]
            idx   = int(label.split()[-1])
            with state_lock:
                current = state["cam_actual"]
            if idx != current:
                with state_lock:
                    state["cam_requested"] = idx
                switch_target = idx
                switching     = True
                window["-CAM_STATUS-"].update("Switching…", text_color="yellow")
            else:
                window["-CAM_STATUS-"].update("Already active", text_color="gray")

        if switching:
            with state_lock:
                actual = state["cam_actual"]
            if actual == switch_target:
                switching = False
                window["-CAM_STATUS-"].update(f"Active: {actual}", text_color="lime")

        with state_lock:
            for k, sk in SLIDER_MAP.items():
                state[sk] = float(values[k])
            state["sprint_enabled"] = bool(values["-SPRINT-"])

            frame       = state["frame"]
            waist_norm  = state["waist_norm"]
            btn_a       = state["btn_a"]
            btn_b       = state["btn_b"]
            rs_click    = state["rs_click"]
            la          = state["left_axis"]
            ra          = state["right_axis"]
            lh_fist     = state["lh_fist"]
            rh_fist     = state["rh_fist"]
            radial_open = state["radial_open"]
            hovered     = state["radial_hovered"]
            combat_mode = state["combat_mode"]

        if frame is not None:
            img_bytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-FRAME-"].update(data=img_bytes)

        w_str = f"{waist_norm:.3f}" if waist_norm is not None else "—"
        window["-FB_WAIST-"].update(f"Waist Y (norm): {w_str}")
        window["-FB_BTNS-"].update(
            f"A: {'■' if btn_a else '□'}   B: {'■' if btn_b else '□'}"
            f"   Sprint RS: {'■' if rs_click else '□'}"
        )
        window["-FB_LAXIS-"].update(f"Left  stick: ({la[0]:+.2f}, {la[1]:+.2f})")
        window["-FB_RAXIS-"].update(f"Right stick: ({ra[0]:+.2f}, {ra[1]:+.2f})")
        window["-FB_FIST-"].update(
            f"LH: {'■ FIST' if lh_fist else '□ open'}"
            f"   RH: {'■ FIST→' + ('RT' if combat_mode=='firing' else 'LT') if rh_fist else '□ open'}"
        )
        window["-FB_MODE-"].update(f"Combat mode: {combat_mode.upper()}")
        hover_label = RADIAL_LABELS[hovered] if hovered >= 0 else "—"
        window["-FB_MENU-"].update(
            f"Radial: {'OPEN – ' + hover_label if radial_open else 'closed'}"
        )

    window.close()

# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not CONTROLLER_AVAILABLE:
        sg.popup_ok(
            "vgamepad not installed.\n\nRun:  pip install vgamepad\n\n"
            "Running in preview-only mode for now.",
            title="Controller unavailable",
        )

    print("[INFO] Enumerating cameras (this may take a moment)...")
    cam_list = enumerate_cameras()
    print(f"[INFO] Found cameras: {cam_list}")

    default_cam = 1 if 1 in cam_list else cam_list[0]
    with state_lock:
        state["cam_index"]  = default_cam
        state["cam_actual"] = default_cam

    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    run_gui(cam_list)

    with state_lock:
        state["running"] = False
    t.join(timeout=3)
    print("Exited cleanly.")