from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import io
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Persistent drawing state (single-user prototype).
draw_canvas = None
prev_point = None
BRUSH_COLOR = (255, 255, 255)
BRUSH_THICKNESS = 5
ERASER_THICKNESS = 30
ERASER_CURSOR_RADIUS = 18
PINCH_LIFT_THRESHOLD = 40
current_tool = "draw"
fist_latched = False


def is_finger_up(landmarks, tip_idx: int, pip_idx: int) -> bool:
    # In image coordinates, smaller y means higher on screen.
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_fist_closed(landmarks) -> bool:
    index_up = is_finger_up(landmarks, tip_idx=8, pip_idx=6)
    middle_up = is_finger_up(landmarks, tip_idx=12, pip_idx=10)
    ring_up = is_finger_up(landmarks, tip_idx=16, pip_idx=14)
    pinky_up = is_finger_up(landmarks, tip_idx=20, pip_idx=18)
    return not (index_up or middle_up or ring_up or pinky_up)


def draw_tool_hud(panel, active_tool: str, cursor_point):
    panel_h, panel_w = panel.shape[:2]
    hud_w, hud_h = 280, 110
    hud_x, hud_y = panel_w - hud_w - 10, 10

    # Background card.
    cv2.rectangle(panel, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (30, 30, 30), -1)
    cv2.rectangle(panel, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (120, 120, 120), 1)

    btn_y1, btn_y2 = hud_y + 10, hud_y + 50
    draw_color = (60, 170, 60) if active_tool == "draw" else (70, 70, 70)
    erase_color = (0, 170, 220) if active_tool == "eraser" else (70, 70, 70)

    cv2.rectangle(panel, (hud_x + 10, btn_y1), (hud_x + 130, btn_y2), draw_color, -1)
    cv2.rectangle(panel, (hud_x + 150, btn_y1), (hud_x + 270, btn_y2), erase_color, -1)
    cv2.putText(panel, "PEN", (hud_x + 46, btn_y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, "ERASER", (hud_x + 165, btn_y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    if cursor_point is not None:
        cx, cy = cursor_point
        cv2.putText(
            panel,
            f"POS: ({cx}, {cy})",
            (hud_x + 12, hud_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            panel,
            "POS: (--, --)",
            (hud_x + 12, hud_y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )


def draw_cursor_preview(panel, cursor_point, tool: str):
    if cursor_point is None:
        return

    cx, cy = cursor_point
    # Crosshair for precise position awareness.
    cv2.line(panel, (max(0, cx - 12), cy), (min(panel.shape[1] - 1, cx + 12), cy), (120, 120, 120), 1)
    cv2.line(panel, (cx, max(0, cy - 12)), (cx, min(panel.shape[0] - 1, cy + 12)), (120, 120, 120), 1)

    if tool == "draw":
        cv2.circle(panel, (cx, cy), 8, (255, 255, 255), 2)
    else:
        cv2.circle(panel, (cx, cy), ERASER_CURSOR_RADIUS, (0, 255, 255), 2)


@app.post("/frame")
async def process_frame(file: UploadFile = File(...)):
    global draw_canvas, prev_point, current_tool, fist_latched

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    frame = cv2.flip(frame, 1)  # mirror view

    h, w, _ = frame.shape

    # Lazily initialize or resize black drawing canvas.
    if draw_canvas is None or draw_canvas.shape[:2] != (h, w):
        draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        prev_point = None

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    status_text = "Move hand into frame"
    cursor_point = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw full hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Index fingertip = landmark 8
            x = int(landmarks[8].x * w)
            y = int(landmarks[8].y * h)
            cursor_point = (x, y)
            thumb_x = int(landmarks[4].x * w)
            thumb_y = int(landmarks[4].y * h)

            # Draw red dot on fingertip
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            fist_closed = is_fist_closed(landmarks)
            if fist_closed and not fist_latched:
                current_tool = "eraser" if current_tool == "draw" else "draw"
                fist_latched = True
                prev_point = None
            elif not fist_closed:
                fist_latched = False

            index_up = is_finger_up(landmarks, tip_idx=8, pip_idx=6)
            middle_up = is_finger_up(landmarks, tip_idx=12, pip_idx=10)
            pinch_distance = int(np.hypot(x - thumb_x, y - thumb_y))
            pen_lifted = pinch_distance < PINCH_LIFT_THRESHOLD
            action_mode = index_up and not middle_up and not fist_closed and not pen_lifted

            if pen_lifted:
                # Visual pinch feedback for pen-up state.
                cv2.line(frame, (x, y), (thumb_x, thumb_y), (255, 200, 0), 2)
                cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 200, 0), -1)

            if current_tool == "eraser":
                # Visible eraser cursor on camera panel.
                cv2.circle(frame, (x, y), ERASER_CURSOR_RADIUS, (0, 255, 255), 2)

            if action_mode:
                if current_tool == "draw":
                    status_text = "TOOL: DRAW | ACTION: DRAWING"
                    if prev_point is not None:
                        cv2.line(draw_canvas, prev_point, (x, y), BRUSH_COLOR, BRUSH_THICKNESS)
                else:
                    status_text = "TOOL: ERASER | ACTION: ERASING"
                    if prev_point is not None:
                        cv2.line(draw_canvas, prev_point, (x, y), (0, 0, 0), ERASER_THICKNESS)
                    # Erase a point even on first contact.
                    cv2.circle(draw_canvas, (x, y), ERASER_THICKNESS // 2, (0, 0, 0), -1)
                prev_point = (x, y)
            else:
                if fist_closed:
                    status_text = f"TOOL: {current_tool.upper()} | FIST DETECTED"
                elif pen_lifted:
                    status_text = f"TOOL: {current_tool.upper()} | PEN UP (PINCH)"
                else:
                    status_text = f"TOOL: {current_tool.upper()} | TRACK MODE"
                prev_point = None

            # Only first hand is used for this step.
            break
    else:
        prev_point = None

    cv2.putText(
        frame,
        status_text,
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Close fist: toggle tool | Pinch: pen up",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw tool HUD and cursor graphics on camera panel.
    draw_tool_hud(frame, current_tool, cursor_point)
    draw_cursor_preview(frame, cursor_point, current_tool)

    # Draw non-destructive cursor/HUD preview on canvas panel.
    canvas_preview = draw_canvas.copy()
    draw_tool_hud(canvas_preview, current_tool, cursor_point)
    draw_cursor_preview(canvas_preview, cursor_point, current_tool)

    # Left: camera frame, Right: black drawing canvas.
    combined = np.hstack([frame, canvas_preview])

    # Encode frame
    _, buffer = cv2.imencode('.jpg', combined)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )
