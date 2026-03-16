'use strict';

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const video        = document.getElementById('video');
const canvas       = document.getElementById('canvas');
const ctx          = canvas.getContext('2d');
const statusPill   = document.getElementById('statusPill');
const toggleBtn    = document.getElementById('toggleBtn');
const hintEl       = document.getElementById('hint');
const gestureLabel = document.getElementById('gestureLabel');
const actionLabel  = document.getElementById('actionLabel');

// ─── Runtime state ────────────────────────────────────────────────────────────
let running        = false;
let handsInstance  = null;
let cameraInstance = null;
let actionTimer    = null;

// Debug: counts frames processed by MediaPipe per second.
let frameCount = 0;
setInterval(() => {
  if (running) {
    const fps = frameCount;
    frameCount = 0;
    hintEl.textContent = `Tracking active — ${fps} fps. Show your hand to the camera.`;
  }
}, 1000);

// Wrist X history for swipe detection across frames (mirrors Python WRIST_HISTORY deque)
const wristHistory    = [];
const MAX_WRIST_HISTORY = 15;

// Cooldowns in milliseconds — mirrors Python can_act() thresholds
const COOLDOWNS = {
  click:    600,
  scroll:   250,
  navigate: 1000,
  tab:      1000,
  close:    1500,
};
const lastActionTime = {};

// EMA cursor smoothing — mirrors Python SMOOTH_ALPHA logic
const SMOOTH_ALPHA = 0.35;
let smoothX    = 0.5;
let smoothY    = 0.5;
let smoothInit = false;

// ─── Cooldown gate ────────────────────────────────────────────────────────────
function canAct(type) {
  const now = Date.now();
  if (!lastActionTime[type] || (now - lastActionTime[type]) > COOLDOWNS[type]) {
    lastActionTime[type] = now;
    return true;
  }
  return false;
}

// ─── Gesture classification (direct port of Python classify_gesture) ──────────
// MediaPipe landmark Y: increases downward. tip.y < pip.y  →  finger is extended.

function isFingerUp(lm, tipIdx, pipIdx) {
  return lm[tipIdx].y < lm[pipIdx].y;
}

function isFistClosed(lm) {
  return !isFingerUp(lm, 8,  6)  &&   // index
         !isFingerUp(lm, 12, 10) &&   // middle
         !isFingerUp(lm, 16, 14) &&   // ring
         !isFingerUp(lm, 20, 18);     // pinky
}

function calcPinchDist(lm) {
  const dx = lm[8].x - lm[4].x;   // index tip – thumb tip
  const dy = lm[8].y - lm[4].y;
  return Math.sqrt(dx * dx + dy * dy);
}

function classifyGesture(lm) {
  if (isFistClosed(lm)) return 'fist';

  if (calcPinchDist(lm) < 0.07) return 'pinch';

  const idx   = isFingerUp(lm, 8,  6);
  const mid   = isFingerUp(lm, 12, 10);
  const ring  = isFingerUp(lm, 16, 14);
  const pinky = isFingerUp(lm, 20, 18);
  const count = [idx, mid, ring, pinky].filter(Boolean).length;

  if (count === 1 && idx)              return 'one_finger';
  if (count === 2 && idx && mid)       return 'two_fingers';
  if (count === 3 && idx && mid && ring) return 'three_fingers';
  if (count === 4)                     return 'open_palm';
  return 'other';
}

// ─── Swipe detection ──────────────────────────────────────────────────────────
// Only called during open_palm frames; history clears when gesture changes.
//
// Note on coordinate orientation:
//   JS canvas is NOT flipped (only the display CSS is mirrored).
//   Original frame: camera's left = x≈0 = user's RIGHT (front-facing camera).
//   User swipes RIGHT  →  x decreases  →  delta < 0  →  go back
//   User swipes LEFT   →  x increases  →  delta > 0  →  go forward
function detectSwipe(lm) {
  wristHistory.push(lm[0].x);
  if (wristHistory.length > MAX_WRIST_HISTORY) wristHistory.shift();
  if (wristHistory.length < 10) return null;

  const delta = wristHistory[wristHistory.length - 1] - wristHistory[0];
  if (delta < -0.15) return 'swipe_right'; // user moved hand right → go back
  if (delta >  0.15) return 'swipe_left';  // user moved hand left  → go forward
  return null;
}

// ─── Flash action label ───────────────────────────────────────────────────────
function flashAction(text) {
  actionLabel.textContent = text;
  actionLabel.classList.add('visible');
  clearTimeout(actionTimer);
  actionTimer = setTimeout(() => actionLabel.classList.remove('visible'), 1400);
}

// ─── Message senders ──────────────────────────────────────────────────────────
function tabAction(action) {
  console.log('[GestureNav] TAB_ACTION:', action);
  chrome.runtime.sendMessage({ type: 'TAB_ACTION', action });
}

function pageAction(action, extra = {}) {
  if (action !== 'MOVE_CURSOR' && action !== 'HIDE_CURSOR') {
    console.log('[GestureNav] PAGE_ACTION:', action, extra);
  }
  chrome.runtime.sendMessage({ type: 'PAGE_ACTION', action, ...extra });
}

// ─── Gesture → action dispatcher ─────────────────────────────────────────────
function handleGesture(gesture, lm) {
  gestureLabel.textContent = gesture.replace(/_/g, ' ');
  console.log('[GestureNav] detected:', gesture);

  // Swipe history is only meaningful during a sustained open_palm.
  // Clear it immediately when any other gesture is detected so stale motion
  // does not accidentally trigger a navigation after the palm is re-raised.
  if (gesture !== 'open_palm') {
    wristHistory.length = 0;
  }

  switch (gesture) {

    // ── One finger: virtual cursor control ─────────────────────────────────
    case 'one_finger': {
      // lm[8] = index fingertip; flip X because the canvas is CSS-mirrored.
      const rawX = 1 - lm[8].x;
      const rawY = lm[8].y;

      if (!smoothInit) {
        smoothX    = rawX;
        smoothY    = rawY;
        smoothInit = true;
      } else {
        smoothX = SMOOTH_ALPHA * rawX + (1 - SMOOTH_ALPHA) * smoothX;
        smoothY = SMOOTH_ALPHA * rawY + (1 - SMOOTH_ALPHA) * smoothY;
      }

      pageAction('MOVE_CURSOR', { nx: smoothX, ny: smoothY });
      break;
    }

    // ── Pinch: click at current virtual cursor position ────────────────────
    case 'pinch': {
      if (canAct('click')) {
        pageAction('CLICK');
        flashAction('Click');
      }
      break;
    }

    // ── Two fingers: scroll direction determined by hand vertical position ──
    case 'two_fingers': {
      if (!canAct('scroll')) break;
      // Wrist Y < 0.5 → hand is in upper half of frame → scroll up.
      if (lm[0].y < 0.5) {
        pageAction('SCROLL_UP');
        flashAction('Scroll up ↑');
      } else {
        pageAction('SCROLL_DOWN');
        flashAction('Scroll down ↓');
      }
      break;
    }

    // ── Three fingers: switch to next tab ──────────────────────────────────
    case 'three_fingers': {
      if (canAct('tab')) {
        tabAction('NEXT_TAB');
        flashAction('Next tab →');
      }
      break;
    }

    // ── Fist: close current tab ────────────────────────────────────────────
    case 'fist': {
      if (canAct('close')) {
        tabAction('CLOSE_TAB');
        flashAction('Close tab ✕');
      }
      break;
    }

    // ── Open palm: idle or swipe for history navigation ────────────────────
    case 'open_palm': {
      // Hide the virtual cursor while the palm is open (user is pausing).
      pageAction('HIDE_CURSOR');
      // Reset cursor smoothing so it snaps cleanly when one_finger resumes.
      smoothInit = false;

      const swipe = detectSwipe(lm);
      if (swipe && canAct('navigate')) {
        if (swipe === 'swipe_right') {
          pageAction('BACK');
          flashAction('← Go back');
        } else {
          pageAction('FORWARD');
          flashAction('Go forward →');
        }
      }
      break;
    }

    default:
      // 'other' — unknown pose; no action.
      break;
  }

  // Keep cursor position alive between one_finger and pinch so the user can
  // aim then click. Reset only when switching to an unrelated gesture.
  if (gesture !== 'one_finger' && gesture !== 'pinch') {
    smoothInit = false;
  }
}

// ─── MediaPipe result callback ────────────────────────────────────────────────
function onHandResults(results) {
  frameCount++;

  // results.image is the <video> element we passed in.
  // .width is the HTML attribute (0 for a hidden video); use .videoWidth for
  // the actual decoded frame resolution. Fall back to the Camera dimensions.
  const w = results.image.videoWidth  || results.image.width  || 320;
  const h = results.image.videoHeight || results.image.height || 240;
  canvas.width  = w;
  canvas.height = h;

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(results.image, 0, 0);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];

    // Draw skeleton overlay.
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color: '#3b82f6', lineWidth: 2 });
    drawLandmarks(ctx,   lm, { color: '#fff', fillColor: '#3b82f6', lineWidth: 1, radius: 4 });

    const gesture = classifyGesture(lm);
    handleGesture(gesture, lm);
  } else {
    // No hand detected — reset everything.
    gestureLabel.textContent = '–';
    wristHistory.length      = 0;
    smoothInit               = false;
    pageAction('HIDE_CURSOR');
  }

  ctx.restore();
}

// ─── Start tracking ───────────────────────────────────────────────────────────
// Asks background to open a small popup window where Chrome reliably shows
// the camera permission dialog. MediaPipe init runs after permission is granted.
function startTracking() {
  setStatus('loading', 'Loading…');
  toggleBtn.disabled = true;
  hintEl.textContent = 'A small popup will open — click Allow to grant camera access.';

  chrome.runtime.sendMessage({ type: 'OPEN_PERMISSION_POPUP' });
}

// ─── Listen for permission result from the popup ──────────────────────────────
chrome.runtime.onMessage.addListener((message) => {
  if (message.type !== 'CAMERA_PERMISSION') return;

  if (message.granted) {
    initMediaPipe();
  } else {
    setStatus('error', 'Camera denied');
    toggleBtn.disabled = false;
    hintEl.textContent = 'Camera access denied. Remove and reload the extension, then click Allow when prompted.';
  }
});

// ─── Initialise MediaPipe after permission is confirmed ───────────────────────
async function initMediaPipe() {
  hintEl.textContent = 'Initialising MediaPipe…';

  try {
    handsInstance = new Hands({
      locateFile: (file) =>
        file.endsWith('.js')
          ? chrome.runtime.getURL(`lib/${file}`)
          : `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });

    handsInstance.setOptions({
      maxNumHands:            1,
      modelComplexity:        1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence:  0.7,
    });

    handsInstance.onResults(onHandResults);

    // Force the WASM binary and TFLite model files to download now.
    // Without this, loading is deferred to the first send() call and any
    // network errors are silently swallowed inside the Camera loop.
    hintEl.textContent = 'Downloading hand detection model (first run may take a moment)…';
    await handsInstance.initialize();
    console.log('[GestureNav] MediaPipe model loaded');

    // Camera utility streams webcam frames to MediaPipe each animation tick.
    cameraInstance = new Camera(video, {
      onFrame: async () => {
        if (!handsInstance) return;
        try {
          await handsInstance.send({ image: video });
        } catch (err) {
          console.error('[GestureNav] send() error:', err);
        }
      },
      width:  320,
      height: 240,
    });

    await cameraInstance.start();

    running = true;
    setStatus('live', 'Live');
    toggleBtn.textContent = 'Stop Gesture Control';
    toggleBtn.className   = 'btn btn--stop';
    toggleBtn.disabled    = false;
    hintEl.textContent    = 'Tracking active. Show your hand to the camera.';

  } catch (err) {
    console.error('[GestureNav]', err);
    setStatus('error', 'Error');
    toggleBtn.disabled = false;
    hintEl.textContent = 'Could not start: ' + err.message;
  }
}

// ─── Stop tracking ────────────────────────────────────────────────────────────
function stopTracking() {
  if (cameraInstance) { cameraInstance.stop(); cameraInstance = null; }
  if (handsInstance)  { handsInstance.close();  handsInstance  = null; }

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  wristHistory.length = 0;
  smoothInit          = false;
  pageAction('DISABLE');

  running = false;
  setStatus('idle', 'Idle');
  toggleBtn.textContent        = 'Start Gesture Control';
  toggleBtn.className          = 'btn btn--start';
  toggleBtn.disabled           = false;
  gestureLabel.textContent     = '–';
  hintEl.textContent           = 'Click Start to enable webcam and begin tracking.';
  actionLabel.classList.remove('visible');
}

// ─── Status pill helper ───────────────────────────────────────────────────────
function setStatus(state, label) {
  statusPill.className   = `pill pill--${state}`;
  statusPill.textContent = label;
}

// ─── Wire up the toggle button ────────────────────────────────────────────────
toggleBtn.addEventListener('click', () => {
  if (running) stopTracking();
  else         startTracking();
});
