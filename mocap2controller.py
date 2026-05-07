"""
Motion Controller
-----------------
Maps body landmarks (via MediaPipe HolisticLandmarker) to a virtual Xbox controller (via vgamepad).
Uses OpenCV for webcam capture and PySimpleGUI for live config.

Controls:
  - Waist Y > upper_bound  → A button
  - Waist Y < lower_bound  → B button
  - Left  hand (x,y) relative to left-stick  center → Left  stick axes
  - Right hand (x,y) relative to right-stick center → Right stick axes
  - Right hand distance > right-stick radius → Right stick click (sprint, toggleable)

Requirements:
    pip install opencv-python mediapipe vgamepad PySimpleGUI numpy requests

On first run, the script will automatically download the holistic_landmarker.task model
from Google's servers (~4 MB) and save it next to the script.

vgamepad will also prompt you to install the ViGEmBus driver on first use.
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
    """Probe indices 0..max_index-1 and return list of (index, label) for working ones."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW is faster on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
            cap.release()
    if not found:
        found = [0]  # fallback
    return found


# ─────────────────────────────────────────────
#  Defaults (all positions are 0‥1 normalised)
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
    lh_fist=False,        # left hand closed  → grenade (LB)
    rh_fist=False,        # right hand closed → trigger (RT)
    cam_width=640,
    cam_height=480,
    cam_fps=30,
    cam_exposure=-5,
    # Camera switching: GUI writes cam_requested, thread reads + clears it
    cam_index=1,
    cam_requested=None,   # set to an int to trigger a switch
    cam_actual=1,         # what's actually open right now (read-only from thread)
))


# ─────────────────────────────────────────────
#  Helper: map hand→stick axis (-1 … +1)
# ─────────────────────────────────────────────
def hand_to_axis(hand_xy, center_xy, radius):
    dx = hand_xy[0] - center_xy[0]
    dy = hand_xy[1] - center_xy[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, 0.0
    scale = min(dist / radius, 1.0)
    angle = math.atan2(dy, dx)
    ax = math.cos(angle) * scale
    ay = -math.sin(angle) * scale
    return ax, ay


# ─────────────────────────────────────────────
#  Fist detection
# ─────────────────────────────────────────────
# MediaPipe hand landmark indices
# Fingertips:  4 (thumb), 8 (index), 12 (middle), 16 (ring), 20 (pinky)
# Mid-knuckles: 3, 7, 11, 15, 19
# Wrist: 0
_TIPS = [8, 12, 16, 20]       # index→pinky tips (skip thumb – unreliable for fist)
_MIDS = [6, 10, 14, 18]       # pip (proximal inter-phalangeal) joints for same fingers

def is_fist(landmarks, threshold=0.8):
    """
    Return True if the hand looks closed.

    Strategy: for each of the four fingers, compare the distance from the
    fingertip to the wrist against the distance from the pip-joint to the
    wrist.  When the finger is curled the tip is CLOSER to the wrist than
    the pip joint is.  We call the hand a fist when the majority of fingers
    satisfy that condition by a comfortable margin (threshold < 1.0 means
    tip must be closer than threshold * pip_dist).
    """
    if not landmarks or len(landmarks) < 21:
        return False

    wrist = landmarks[0]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    curled = 0
    for tip_idx, mid_idx in zip(_TIPS, _MIDS):
        tip_dist = dist(landmarks[tip_idx], wrist)
        pip_dist = dist(landmarks[mid_idx], wrist)
        if pip_dist > 0 and tip_dist < pip_dist * threshold:
            curled += 1

    return curled >= 3   # at least 3 of 4 fingers curled


# ─────────────────────────────────────────────
#  Controller wrapper (vgamepad)
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
                   dot_color=(0, 255, 0), line_color=(0, 200, 0),
                   dot_r=4, thickness=2):
    if not landmarks:
        return
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], line_color, thickness)
    for pt in pts:
        cv2.circle(frame, pt, dot_r, dot_color, -1)


# ─────────────────────────────────────────────
#  Capture + detection thread
# ─────────────────────────────────────────────
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cap


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

    BTN_A  = vg.XUSB_BUTTON.XUSB_GAMEPAD_A           if CONTROLLER_AVAILABLE else None
    BTN_B  = vg.XUSB_BUTTON.XUSB_GAMEPAD_B           if CONTROLLER_AVAILABLE else None
    BTN_RS = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB if CONTROLLER_AVAILABLE else None

    frame_ts = 0

    with HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            # ── Check running + camera switch request ──────────────────
            with state_lock:
                if not state["running"]:
                    break
                requested = state["cam_requested"]
                if requested is not None and requested != current_index:
                    state["cam_requested"] = None   # consume the request
                else:
                    requested = None

            if requested is not None:
                print(f"[INFO] Switching camera: {current_index} → {requested}")
                cap.release()
                cap = open_camera(requested)
                current_index = requested
                with state_lock:
                    state["cam_actual"] = current_index
                # Show a blank frame while switching
                with state_lock:
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
                wu        = state["waist_upper"]
                wl        = state["waist_lower"]
                lcx       = state["left_cx"];  lcy = state["left_cy"];  lr = state["left_r"]
                rcx       = state["right_cx"]; rcy = state["right_cy"]; rr = state["right_r"]
                sprint_on = state["sprint_enabled"]

            # Pose
            waist_norm = None
            pose_lms = result.pose_landmarks
            if pose_lms and len(pose_lms) > 24:
                lhip = pose_lms[23]
                rhip = pose_lms[24]
                waist_norm = (lhip.y + rhip.y) / 2.0

            # Hands (mirrored frame: mediapipe's "left" is our right, vice versa)
            rh_lms = result.left_hand_landmarks   # right hand landmarks (mirrored)
            lh_lms = result.right_hand_landmarks  # left  hand landmarks (mirrored)

            rh_norm = None
            if rh_lms:
                rh_norm = (rh_lms[0].x, rh_lms[0].y)

            lh_norm = None
            if lh_lms:
                lh_norm = (lh_lms[0].x, lh_lms[0].y)

            # Fist detection
            rh_fist = is_fist(rh_lms)   # right fist → hold RT (trigger)
            lh_fist = is_fist(lh_lms)   # left  fist → LB (grenade)

            # Logic
            btn_a = bool(waist_norm is not None and waist_norm < wu)
            btn_b = bool(waist_norm is not None and waist_norm > wl)

            left_axis  = hand_to_axis(lh_norm, (lcx, lcy), lr) if lh_norm else (0.0, 0.0)
            right_axis = hand_to_axis(rh_norm, (rcx, rcy), rr) if rh_norm else (0.0, 0.0)

            rs_click = False
            if sprint_on and rh_norm:
                dist = math.hypot(rh_norm[0] - rcx, rh_norm[1] - rcy)
                rs_click = dist > rr

            ctrl.set_button(BTN_A,  btn_a)
            ctrl.set_button(BTN_B,  btn_b)
            ctrl.set_button(BTN_RS, rs_click)
            # Both triggers are axes (0.0–1.0), not buttons
            if CONTROLLER_AVAILABLE and ctrl.gamepad:
                try:
                    ctrl.gamepad.left_trigger_float(value_float=1.0 if lh_fist else 0.0)
                    ctrl.gamepad.right_trigger_float(value_float=1.0 if rh_fist else 0.0)
                except Exception:
                    pass
            ctrl.set_axes(left_axis[0], left_axis[1], right_axis[0], right_axis[1])
            ctrl.flush()

            # ── Draw overlays ──────────────────────────────────────────
            if pose_lms:
                draw_landmarks(frame, pose_lms, POSE_CONNECTIONS, h, w,
                               dot_color=(66, 117, 245), line_color=(230, 66, 245))
            if lh_lms:
                draw_landmarks(frame, lh_lms,
                               HAND_CONNECTIONS, h, w,
                               dot_color=(0, 200, 255), line_color=(0, 150, 200))
            if rh_lms:
                draw_landmarks(frame, rh_lms,
                               HAND_CONNECTIONS, h, w,
                               dot_color=(255, 200, 0), line_color=(200, 150, 0))

            # Waist bounds
            cv2.line(frame, (0, int(wu * h)), (w, int(wu * h)), (0, 255, 0), 2)
            cv2.putText(frame, "A (raise above)", (5, int(wu * h) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.line(frame, (0, int(wl * h)), (w, int(wl * h)), (0, 120, 255), 2)
            cv2.putText(frame, "B (crouch below)", (5, int(wl * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

            if waist_norm is not None:
                wy = int(waist_norm * h)
                cv2.line(frame, (0, wy), (w, wy), (255, 255, 0), 2)
                cv2.putText(frame, "Waist", (5, wy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Stick circles
            lc_px = (int(lcx * w), int(lcy * h))
            rc_px = (int(rcx * w), int(rcy * h))
            cv2.circle(frame, lc_px, int(lr * w), (0, 200, 255), 2)
            cv2.circle(frame, rc_px, int(rr * w),
                       (0, 60, 255) if sprint_on else (160, 160, 160), 2)
            cv2.putText(frame, "L", (lc_px[0] - 6, lc_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(frame, "R", (rc_px[0] - 6, rc_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 2)

            if lh_norm:
                lh_px = (int(lh_norm[0] * w), int(lh_norm[1] * h))
                lh_col = (0, 0, 220) if lh_fist else (0, 200, 255)   # red tint when fist
                cv2.circle(frame, lh_px, 12, lh_col, -1)
                cv2.putText(frame, "FIST" if lh_fist else "open",
                            (lh_px[0] + 14, lh_px[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 80, 255) if lh_fist else (0, 200, 255), 1)
                cv2.line(frame, lh_px, lc_px, (0, 200, 255), 2)
            if rh_norm:
                rh_px = (int(rh_norm[0] * w), int(rh_norm[1] * h))
                rh_col = (0, 0, 220) if rh_fist else (0, 60, 255)    # red tint when fist
                if rs_click:
                    rh_col = (0, 0, 180)
                cv2.circle(frame, rh_px, 12, rh_col, -1)
                cv2.putText(frame, "FIST" if rh_fist else "open",
                            (rh_px[0] + 14, rh_px[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255) if rh_fist else (0, 60, 255), 1)
                cv2.line(frame, rh_px, rc_px, (0, 60, 255), 2)

            # Camera index badge (top-right)
            badge = f"CAM {current_index}"
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (w - bw - 12, 4), (w - 4, bh + 10), (30, 30, 30), -1)
            cv2.putText(frame, badge, (w - bw - 8, bh + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)

            # HUD
            hud = [
                f"A: {'ON' if btn_a else 'off'}   B: {'ON' if btn_b else 'off'}"
                f"   Sprint RS: {'ON' if rs_click else 'off'}",
                f"L ({left_axis[0]:+.2f}, {left_axis[1]:+.2f})"
                f"   R ({right_axis[0]:+.2f}, {right_axis[1]:+.2f})",
                f"LH: {'FIST→grenade(LT)' if lh_fist else 'open'}"
                f"   RH: {'FIST→trigger(RT)' if rh_fist else 'open'}",
            ]
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (8, 22 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                            cv2.LINE_AA)

            disp_w = min(w, 640)
            disp_h = int(h * disp_w / w)
            display = cv2.resize(frame, (disp_w, disp_h))

            with state_lock:
                state["frame"]      = display
                state["waist_norm"] = waist_norm
                state["lh_norm"]    = lh_norm
                state["rh_norm"]    = rh_norm
                state["btn_a"]      = btn_a
                state["btn_b"]      = btn_b
                state["rs_click"]   = rs_click
                state["left_axis"]  = left_axis
                state["right_axis"] = right_axis
                state["lh_fist"]    = lh_fist
                state["rh_fist"]    = rh_fist

    ctrl.reset()
    cap.release()


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
def slider(key, label, lo, hi, default, resolution=0.01):
    return [
        sg.Text(label, size=(24, 1)),
        sg.Slider(range=(lo, hi), default_value=default, resolution=resolution,
                  orientation="h", size=(28, 15), key=key, enable_events=True),
    ]


def build_layout(cam_list, initial_cam):
    frame_col = [[sg.Image(key="-FRAME-", size=(640, 480))]]

    cam_labels = [f"Camera {i}" for i in cam_list]
    initial_label = f"Camera {initial_cam}" if initial_cam in cam_list else cam_labels[0]

    controls_col = [
        [sg.Frame("Camera", [
            [
                sg.Text("Source:", size=(8, 1)),
                sg.Combo(cam_labels, default_value=initial_label,
                         key="-CAM_SELECT-", size=(14, 1), readonly=True,
                         enable_events=True),
                sg.Button("Switch", key="-CAM_SWITCH-", size=(7, 1)),
                sg.Text("", key="-CAM_STATUS-", size=(14, 1), text_color="yellow"),
            ],
        ])],
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
            [sg.Text("", key="-FB_WAIST-", size=(44, 1))],
            [sg.Text("", key="-FB_BTNS-",  size=(44, 1))],
            [sg.Text("", key="-FB_LAXIS-", size=(44, 1))],
            [sg.Text("", key="-FB_RAXIS-", size=(44, 1))],
            [sg.Text("", key="-FB_FIST-",  size=(44, 1))],
        ])],
        [sg.Button("Reset Defaults", key="-RESET-"),
         sg.Button("Quit",           key="-QUIT-")],
    ]

    return [
        [
            sg.Column(frame_col,    vertical_alignment="top"),
            sg.VSeparator(),
            sg.Column(controls_col, vertical_alignment="top"),
        ]
    ]


def apply_defaults(window):
    window["-WAIST_UP-"].update(DEFAULT["waist_upper"])
    window["-WAIST_LOW-"].update(DEFAULT["waist_lower"])
    window["-LCX-"].update(DEFAULT["left_cx"])
    window["-LCY-"].update(DEFAULT["left_cy"])
    window["-LR-"].update(DEFAULT["left_r"])
    window["-RCX-"].update(DEFAULT["right_cx"])
    window["-RCY-"].update(DEFAULT["right_cy"])
    window["-RR-"].update(DEFAULT["right_r"])
    window["-SPRINT-"].update(DEFAULT["sprint_enabled"])


SLIDER_MAP = {
    "-WAIST_UP-":  "waist_upper",
    "-WAIST_LOW-": "waist_lower",
    "-LCX-":       "left_cx",
    "-LCY-":       "left_cy",
    "-LR-":        "left_r",
    "-RCX-":       "right_cx",
    "-RCY-":       "right_cy",
    "-RR-":        "right_r",
}


def run_gui(cam_list):
    sg.theme("DarkBlue3")

    with state_lock:
        initial_cam = state["cam_index"]

    window = sg.Window(
        "Motion Controller",
        build_layout(cam_list, initial_cam),
        finalize=True,
    )

    switching = False       # True while we're waiting for the thread to confirm
    switch_target = None

    while True:
        event, values = window.read(timeout=30)

        if event in (sg.WIN_CLOSED, "-QUIT-"):
            with state_lock:
                state["running"] = False
            break

        if event == "-RESET-":
            apply_defaults(window)

        # Camera switch: request it, show pending status
        if event == "-CAM_SWITCH-":
            label = values["-CAM_SELECT-"]           # e.g. "Camera 2"
            idx = int(label.split()[-1])
            with state_lock:
                current = state["cam_actual"]
            if idx != current:
                with state_lock:
                    state["cam_requested"] = idx
                switch_target = idx
                switching = True
                window["-CAM_STATUS-"].update("Switching…", text_color="yellow")
            else:
                window["-CAM_STATUS-"].update("Already active", text_color="gray")

        # Poll for switch completion
        if switching:
            with state_lock:
                actual = state["cam_actual"]
            if actual == switch_target:
                switching = False
                window["-CAM_STATUS-"].update(f"Active: {actual}", text_color="lime")

        # Sync sliders
        with state_lock:
            for k, sk in SLIDER_MAP.items():
                state[sk] = float(values[k])
            state["sprint_enabled"] = bool(values["-SPRINT-"])

            frame      = state["frame"]
            waist_norm = state["waist_norm"]
            btn_a      = state["btn_a"]
            btn_b      = state["btn_b"]
            rs_click   = state["rs_click"]
            la         = state["left_axis"]
            ra         = state["right_axis"]
            lh_fist    = state["lh_fist"]
            rh_fist    = state["rh_fist"]

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
            f"LH: {'■ FIST→LT(grenade)' if lh_fist else '□ open'}"
            f"   RH: {'■ FIST→RT(trigger)' if rh_fist else '□ open'}"
        )

    window.close()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not CONTROLLER_AVAILABLE:
        sg.popup_ok(
            "vgamepad not installed.\n\n"
            "Run:  pip install vgamepad\n\n"
            "The pip install will prompt you to install the ViGEmBus driver automatically.\n"
            "Running in preview-only mode for now.",
            title="Controller unavailable",
        )

    print("[INFO] Enumerating cameras (this may take a moment)...")
    cam_list = enumerate_cameras()
    print(f"[INFO] Found cameras: {cam_list}")

    # Default to camera 1 if available, otherwise first found
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