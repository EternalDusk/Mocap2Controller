"""
Motion Controller
-----------------
Maps body landmarks (via MediaPipe Holistic) to a virtual Xbox controller (via pyXInput).
Uses OpenCV for webcam capture and PySimpleGUI for live config.

Controls:
  - Waist Y > upper_bound  → A button
  - Waist Y < lower_bound  → B button
  - Left  hand (x,y) relative to left-stick  center → Left  stick axes
  - Right hand (x,y) relative to right-stick center → Right stick axes
  - Right hand distance > right-stick radius → Right stick click (sprint, toggleable)

Requirements:
    pip install opencv-python mediapipe pyxinput PySimpleGUI numpy
"""

import math
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import PySimpleGUI as sg

# pyXInput ships as "pyXInput" but is imported as pyxinput (lowercase)
try:
    import pyxinput
    CONTROLLER_AVAILABLE = True
except ImportError:
    CONTROLLER_AVAILABLE = False
    print("[WARN] pyxinput not found – running in preview-only mode.")

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
#  Shared state (read by GUI thread, written by
#  capture thread and vice-versa via lock)
# ─────────────────────────────────────────────
state_lock = threading.Lock()
state = dict(DEFAULT)
state.update(dict(
    frame=None,          # latest annotated BGR frame
    running=True,
    # live feedback
    waist_norm=None,     # 0‥1
    lh_norm=None,        # (x, y) 0‥1
    rh_norm=None,
    btn_a=False,
    btn_b=False,
    rs_click=False,
    left_axis=(0.0, 0.0),
    right_axis=(0.0, 0.0),
))


# ─────────────────────────────────────────────
#  Helper: map hand→stick axis (-1 … +1)
# ─────────────────────────────────────────────
def hand_to_axis(hand_xy, center_xy, radius):
    """
    Returns (x_axis, y_axis) in [-1, 1].
    Clamps magnitude to 1.  Y is flipped so up = positive.
    """
    dx = hand_xy[0] - center_xy[0]
    dy = hand_xy[1] - center_xy[1]   # screen Y increases downward
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, 0.0
    scale = min(dist / radius, 1.0)
    angle = math.atan2(dy, dx)
    ax = math.cos(angle) * scale
    ay = -math.sin(angle) * scale    # invert Y for stick convention
    return ax, ay


# ─────────────────────────────────────────────
#  Controller wrapper
# ─────────────────────────────────────────────
class VirtualController:
    def __init__(self):
        self.ctrl = None
        if CONTROLLER_AVAILABLE:
            try:
                self.ctrl = pyxinput.vController()
                print("[INFO] Virtual controller created.")
            except Exception as e:
                print(f"[WARN] Could not create virtual controller: {e}")

    def set_button(self, name, value):
        if self.ctrl:
            try:
                self.ctrl.set_value(name, int(bool(value)))
            except Exception:
                pass

    def set_axis(self, name, value):
        """value: -1.0 … 1.0"""
        if self.ctrl:
            try:
                # pyXInput expects -32768 … 32767
                raw = int(max(-1.0, min(1.0, value)) * 32767)
                self.ctrl.set_value(name, raw)
            except Exception:
                pass

    def reset(self):
        for btn in ("BtnA", "BtnB", "BtnThumbR"):
            self.set_button(btn, False)
        for ax in ("AxisLx", "AxisLy", "AxisRx", "AxisRy"):
            self.set_axis(ax, 0.0)


# ─────────────────────────────────────────────
#  Capture + detection thread
# ─────────────────────────────────────────────
def capture_thread():
    mp_holistic = mp.solutions.holistic
    mp_drawing  = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    ctrl = VirtualController()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while True:
            with state_lock:
                if not state["running"]:
                    break

            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)   # mirror for intuitive feel
            h, w = frame.shape[:2]

            # ── MediaPipe ──────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # ── Read config snapshot ───────────────────────
            with state_lock:
                wu = state["waist_upper"]
                wl = state["waist_lower"]
                lcx, lcy, lr = state["left_cx"],  state["left_cy"],  state["left_r"]
                rcx, rcy, rr = state["right_cx"], state["right_cy"], state["right_r"]
                sprint_on    = state["sprint_enabled"]

            # ── Waist detection (mid-point of hips) ────────
            waist_norm = None
            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark
                lhip = lms[mp_holistic.PoseLandmark.LEFT_HIP]
                rhip = lms[mp_holistic.PoseLandmark.RIGHT_HIP]
                waist_norm = (lhip.y + rhip.y) / 2.0   # 0 = top, 1 = bottom

            # ── Hand detection ─────────────────────────────
            lh_norm = None
            if results.left_hand_landmarks:
                lms = results.left_hand_landmarks.landmark
                # Use wrist (landmark 0) as hand position
                lh_norm = (lms[0].x, lms[0].y)

            rh_norm = None
            if results.right_hand_landmarks:
                lms = results.right_hand_landmarks.landmark
                rh_norm = (lms[0].x, lms[0].y)

            # ── Button / axis logic ────────────────────────
            btn_a = bool(waist_norm is not None and waist_norm < wu)
            btn_b = bool(waist_norm is not None and waist_norm > wl)

            left_axis  = hand_to_axis(lh_norm, (lcx, lcy), lr) if lh_norm else (0.0, 0.0)
            right_axis = hand_to_axis(rh_norm, (rcx, rcy), rr) if rh_norm else (0.0, 0.0)

            rs_click = False
            if sprint_on and rh_norm:
                dist = math.hypot(rh_norm[0] - rcx, rh_norm[1] - rcy)
                rs_click = dist > rr

            # ── Send to controller ─────────────────────────
            ctrl.set_button("BtnA",      btn_a)
            ctrl.set_button("BtnB",      btn_b)
            ctrl.set_button("BtnThumbR", rs_click)
            ctrl.set_axis("AxisLx", left_axis[0])
            ctrl.set_axis("AxisLy", left_axis[1])
            ctrl.set_axis("AxisRx", right_axis[0])
            ctrl.set_axis("AxisRy", right_axis[1])

            # ── Draw overlays ──────────────────────────────
            # Pose skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2),
                )

            # Hand skeletons
            for hand_lms, conn in [
                (results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS),
                (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
            ]:
                if hand_lms:
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, conn,
                        mp_drawing.DrawingSpec(
                            color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(
                            color=(121, 44, 250), thickness=2),
                    )

            # Waist bounds lines
            cv2.line(frame,
                     (0, int(wu * h)), (w, int(wu * h)),
                     (0, 255, 0), 2)
            cv2.putText(frame, "A", (5, int(wu * h) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(frame,
                     (0, int(wl * h)), (w, int(wl * h)),
                     (0, 100, 255), 2)
            cv2.putText(frame, "B", (5, int(wl * h) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

            # Stick circles
            lc_px = (int(lcx * w), int(lcy * h))
            rc_px = (int(rcx * w), int(rcy * h))
            lr_px = int(lr * w)
            rr_px = int(rr * w)

            cv2.circle(frame, lc_px, lr_px, (255, 200, 0), 2)
            cv2.circle(frame, rc_px, rr_px,
                       (0, 80, 255) if sprint_on else (180, 180, 180), 2)
            cv2.putText(frame, "L", (lc_px[0] - 6, lc_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            cv2.putText(frame, "R", (rc_px[0] - 6, rc_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2)

            # Hand dots
            if lh_norm:
                cv2.circle(frame,
                           (int(lh_norm[0] * w), int(lh_norm[1] * h)),
                           10, (255, 200, 0), -1)
            if rh_norm:
                col = (0, 0, 255) if rs_click else (0, 80, 255)
                cv2.circle(frame,
                           (int(rh_norm[0] * w), int(rh_norm[1] * h)),
                           10, col, -1)

            # HUD
            hud_lines = [
                f"A: {'ON' if btn_a else 'off'}   B: {'ON' if btn_b else 'off'}   "
                f"RS: {'SPRINT' if rs_click else 'off'}",
                f"L-axis ({left_axis[0]:+.2f}, {left_axis[1]:+.2f})   "
                f"R-axis ({right_axis[0]:+.2f}, {right_axis[1]:+.2f})",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(frame, line, (8, 22 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                            cv2.LINE_AA)

            # ── Publish frame + feedback ───────────────────
            # Resize for GUI embed (keep aspect, cap at 640 wide)
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
        sg.Text(label, size=(22, 1)),
        sg.Slider(range=(lo, hi), default_value=default, resolution=resolution,
                  orientation="h", size=(28, 15), key=key,
                  enable_events=True),
    ]


def build_layout():
    frame_col = [[sg.Image(key="-FRAME-", size=(640, 480))]]

    controls_col = [
        [sg.Frame("Waist Bounds", [
            slider("-WAIST_UP-",  "Upper (A above)",  0.0, 1.0, DEFAULT["waist_upper"]),
            slider("-WAIST_LOW-", "Lower (B below)",  0.0, 1.0, DEFAULT["waist_lower"]),
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
            [sg.Checkbox("Enable sprint on right-hand reach",
                         default=DEFAULT["sprint_enabled"],
                         key="-SPRINT-", enable_events=True)],
        ])],
        [sg.HSeparator()],
        [sg.Frame("Live Feedback", [
            [sg.Text("", key="-FB_WAIST-",  size=(40, 1))],
            [sg.Text("", key="-FB_BTNS-",   size=(40, 1))],
            [sg.Text("", key="-FB_LAXIS-",  size=(40, 1))],
            [sg.Text("", key="-FB_RAXIS-",  size=(40, 1))],
        ])],
        [sg.Button("Reset Defaults", key="-RESET-"),
         sg.Button("Quit",           key="-QUIT-")],
    ]

    return [
        [
            sg.Column(frame_col,    vertical_alignment="top"),
            sg.VSeparator(),
            sg.Column(controls_col, vertical_alignment="top", scrollable=False),
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


def run_gui():
    sg.theme("DarkBlue3")
    window = sg.Window(
        "Motion Controller",
        build_layout(),
        finalize=True,
        return_keyboard_events=False,
    )

    # Map slider keys → state keys
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

    while True:
        event, values = window.read(timeout=30)   # ~33 fps

        if event in (sg.WIN_CLOSED, "-QUIT-"):
            with state_lock:
                state["running"] = False
            break

        if event == "-RESET-":
            apply_defaults(window)
            with state_lock:
                state.update({v: DEFAULT[v] for v in DEFAULT})

        # Slider / checkbox events → update shared state
        if event in SLIDER_MAP:
            with state_lock:
                state[SLIDER_MAP[event]] = float(values[event])

        if event == "-SPRINT-":
            with state_lock:
                state["sprint_enabled"] = bool(values["-SPRINT-"])

        # Always sync all sliders (handles programmatic updates too)
        with state_lock:
            for k, sk in SLIDER_MAP.items():
                state[sk] = float(values[k])
            state["sprint_enabled"] = bool(values["-SPRINT-"])

            # Grab display snapshot
            frame      = state["frame"]
            waist_norm = state["waist_norm"]
            btn_a      = state["btn_a"]
            btn_b      = state["btn_b"]
            rs_click   = state["rs_click"]
            la         = state["left_axis"]
            ra         = state["right_axis"]

        # Update camera frame
        if frame is not None:
            img_bytes = cv2.imencode(".png", frame)[1].tobytes()
            window["-FRAME-"].update(data=img_bytes)

        # Update feedback text
        w_str = f"{waist_norm:.3f}" if waist_norm is not None else "—"
        window["-FB_WAIST-"].update(f"Waist Y: {w_str}")
        window["-FB_BTNS-"].update(
            f"A: {'■' if btn_a else '□'}   B: {'■' if btn_b else '□'}   "
            f"Sprint: {'■' if rs_click else '□'}"
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
            "pyxinput not installed or ScpVBus driver not present.\n"
            "Running in preview-only mode (no controller output).",
            title="Controller unavailable",
        )

    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    run_gui()

    with state_lock:
        state["running"] = False
    t.join(timeout=3)
    print("Exited cleanly.")