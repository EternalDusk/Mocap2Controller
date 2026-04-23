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
        print(f"[INFO] Downloading holistic_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved to {MODEL_PATH}")


# ─────────────────────────────────────────────
#  Defaults (all positions are 0‥1 normalised)
# ─────────────────────────────────────────────
DEFAULT = dict(
    waist_upper=0.35,   # normalised Y – above this → A
    waist_lower=0.65,   # normalised Y – below this → B

    left_cx=0.25,       # left-stick  circle centre X
    left_cy=0.50,       # left-stick  circle centre Y
    left_r=0.15,        # left-stick  circle radius (as fraction of frame width)

    right_cx=0.75,      # right-stick circle centre X
    right_cy=0.50,      # right-stick circle centre Y
    right_r=0.15,       # right-stick circle radius

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
    cam_width=640,
    cam_height=480,
    cam_fps=30,
    cam_exposure=-5,  # typical default range, varies by camera
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
    ay = -math.sin(angle) * scale   # invert Y: screen-up = stick-up
    return ax, ay



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
#  Drawing helpers (new API has no draw_landmarks for holistic directly)
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
    """Draw landmark dots and connection lines from a flat NormalizedLandmark list."""
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
def capture_thread():
    ensure_model()

    # New tasks API imports
    BaseOptions          = mp.tasks.BaseOptions
    HolisticLandmarker  = mp.tasks.vision.HolisticLandmarker
    HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    cap = cv2.VideoCapture(1)

    ctrl = VirtualController()

    BTN_A  = vg.XUSB_BUTTON.XUSB_GAMEPAD_A           if CONTROLLER_AVAILABLE else None
    BTN_B  = vg.XUSB_BUTTON.XUSB_GAMEPAD_B           if CONTROLLER_AVAILABLE else None
    BTN_RS = vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB if CONTROLLER_AVAILABLE else None

    frame_ts = 0  # running millisecond timestamp for VIDEO mode

    with HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            with state_lock:
                if not state["running"]:
                    break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)   # mirror – also fixes left/right hand labels
            h, w = frame.shape[:2]

            # Wrap frame for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs a monotonically increasing timestamp in ms
            frame_ts += 33
            result = landmarker.detect_for_video(mp_image, frame_ts)

            # Read config snapshot
            with state_lock:
                wu  = state["waist_upper"]
                wl  = state["waist_lower"]
                lcx = state["left_cx"];  lcy = state["left_cy"];  lr = state["left_r"]
                rcx = state["right_cx"]; rcy = state["right_cy"]; rr = state["right_r"]
                sprint_on = state["sprint_enabled"]

            # ── Extract pose landmarks (waist = mid-hips) ─────────────
            # result.pose_landmarks is a flat List[NormalizedLandmark] (33 entries),
            # or an empty list if no pose detected.
            # Landmark indices: 23=left hip, 24=right hip
            waist_norm = None
            pose_lms = result.pose_landmarks
            if pose_lms and len(pose_lms) > 24:
                lhip = pose_lms[23]
                rhip = pose_lms[24]
                waist_norm = (lhip.y + rhip.y) / 2.0

            # ── Extract hand landmarks ─────────────────────────────────
            # result.left_hand_landmarks and result.right_hand_landmarks are each
            # a flat List[NormalizedLandmark] (21 entries), or empty if not detected.
            # Wrist = index 0.
            rh_norm = None
            if result.left_hand_landmarks:
                rh_norm = (result.left_hand_landmarks[0].x,
                           result.left_hand_landmarks[0].y)

            lh_norm = None
            if result.right_hand_landmarks:
                lh_norm = (result.right_hand_landmarks[0].x,
                           result.right_hand_landmarks[0].y)

            # ── Button / axis logic ────────────────────────────────────
            btn_a = bool(waist_norm is not None and waist_norm < wu)
            btn_b = bool(waist_norm is not None and waist_norm > wl)

            left_axis  = hand_to_axis(lh_norm, (lcx, lcy), lr) if lh_norm else (0.0, 0.0)
            right_axis = hand_to_axis(rh_norm, (rcx, rcy), rr) if rh_norm else (0.0, 0.0)

            rs_click = False
            if sprint_on and rh_norm:
                dist = math.hypot(rh_norm[0] - rcx, rh_norm[1] - rcy)
                rs_click = dist > rr

            # Send to controller
            ctrl.set_button(BTN_A,  btn_a)
            ctrl.set_button(BTN_B,  btn_b)
            ctrl.set_button(BTN_RS, rs_click)
            ctrl.set_axes(left_axis[0], left_axis[1],
                          right_axis[0], right_axis[1])
            ctrl.flush()

            # ── Draw overlays ──────────────────────────────────────────
            if pose_lms:
                draw_landmarks(frame, pose_lms, POSE_CONNECTIONS, h, w,
                               dot_color=(66, 117, 245), line_color=(230, 66, 245))

            if result.left_hand_landmarks:
                draw_landmarks(frame, result.left_hand_landmarks,
                               HAND_CONNECTIONS, h, w,
                               dot_color=(0, 200, 255), line_color=(0, 150, 200))

            if result.right_hand_landmarks:
                draw_landmarks(frame, result.right_hand_landmarks,
                               HAND_CONNECTIONS, h, w,
                               dot_color=(255, 200, 0), line_color=(200, 150, 0))

            # Waist bound lines
            cv2.line(frame, (0, int(wu * h)), (w, int(wu * h)), (0, 255, 0), 2)
            cv2.putText(frame, "A (raise above)", (5, int(wu * h) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.line(frame, (0, int(wl * h)), (w, int(wl * h)), (0, 120, 255), 2)
            cv2.putText(frame, "B (crouch below)", (5, int(wl * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 2)

            # Live waist line
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

            # Hand wrist dots
            if lh_norm:
                cv2.circle(frame,
                           (int(lh_norm[0] * w), int(lh_norm[1] * h)),
                           10, (0, 200, 255), -1)
            if rh_norm:
                col = (0, 0, 255) if rs_click else (0, 60, 255)
                cv2.circle(frame,
                           (int(rh_norm[0] * w), int(rh_norm[1] * h)),
                           10, col, -1)

            # Draw lines from hands to stick centers
            if lh_norm:
                lh_px = (int(lh_norm[0] * w), int(lh_norm[1] * h))
                cv2.line(frame, lh_px, lc_px, (0, 200, 255), 2)

            if rh_norm:
                rh_px = (int(rh_norm[0] * w), int(rh_norm[1] * h))
                cv2.line(frame, rh_px, rc_px, (0, 60, 255), 2)

            # HUD
            hud = [
                f"A: {'ON' if btn_a else 'off'}   B: {'ON' if btn_b else 'off'}"
                f"   Sprint RS: {'ON' if rs_click else 'off'}",
                f"L ({left_axis[0]:+.2f}, {left_axis[1]:+.2f})"
                f"   R ({right_axis[0]:+.2f}, {right_axis[1]:+.2f})",
            ]
            for i, line in enumerate(hud):
                cv2.putText(frame, line, (8, 22 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                            cv2.LINE_AA)

            # Publish frame
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


def build_layout():
    frame_col = [[sg.Image(key="-FRAME-", size=(640, 480))]]

    controls_col = [
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


def run_gui():
    sg.theme("DarkBlue3")
    window = sg.Window(
        "Motion Controller",
        build_layout(),
        finalize=True,
    )

    while True:
        event, values = window.read(timeout=30)

        if event in (sg.WIN_CLOSED, "-QUIT-"):
            with state_lock:
                state["running"] = False
            break

        if event == "-RESET-":
            apply_defaults(window)
        
        if event == "-APPLY_CAM-":
            with state_lock:
                state["cam_width"] = int(values["-CAM_W-"])
                state["cam_height"] = int(values["-CAM_H-"])
                state["cam_fps"] = int(values["-CAM_FPS-"])
                state["cam_exposure"] = float(values["-CAM_EXP-"])

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

    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    run_gui()

    with state_lock:
        state["running"] = False
    t.join(timeout=3)
    print("Exited cleanly.")