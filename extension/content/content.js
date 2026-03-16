// ─── GestureNav Content Script ─────────────────────────────────────────────────
// Injected into every page. Handles:
//   - Scroll commands
//   - History navigation (back / forward)
//   - Virtual cursor overlay (moveable dot that tracks finger position)
//   - Simulated click at the cursor position

(function () {
  'use strict';

  // Prevent double-injection if the script is somehow re-run.
  if (window.__gestureNavLoaded) return;
  window.__gestureNavLoaded = true;

  // ── Virtual cursor ────────────────────────────────────────────────────────────
  let cursor = null;
  let cursorX = window.innerWidth / 2;
  let cursorY = window.innerHeight / 2;

  function ensureCursor() {
    if (cursor) return;
    cursor = document.createElement('div');
    cursor.id = '__gesturenav_cursor__';
    cursor.style.cssText = [
      'position:fixed',
      'width:22px',
      'height:22px',
      'background:rgba(59,130,246,0.85)',
      'border:2.5px solid #fff',
      'border-radius:50%',
      'box-shadow:0 0 0 3px rgba(59,130,246,0.3)',
      'pointer-events:none',
      'z-index:2147483647',
      'transform:translate(-50%,-50%)',
      'transition:background 0.15s,box-shadow 0.15s',
      'display:none',
    ].join(';');
    document.documentElement.appendChild(cursor);
  }

  function moveCursor(nx, ny) {
    ensureCursor();
    // nx, ny are normalised [0,1] relative to viewport.
    cursorX = nx * window.innerWidth;
    cursorY = ny * window.innerHeight;
    cursor.style.left    = cursorX + 'px';
    cursor.style.top     = cursorY + 'px';
    cursor.style.display = 'block';
  }

  function hideCursor() {
    if (cursor) cursor.style.display = 'none';
  }

  // ── Simulated click ───────────────────────────────────────────────────────────
  function simulateClick() {
    ensureCursor();

    // Flash the cursor red to give visual feedback.
    cursor.style.background   = 'rgba(239,68,68,0.9)';
    cursor.style.boxShadow    = '0 0 0 5px rgba(239,68,68,0.35)';
    setTimeout(() => {
      if (cursor) {
        cursor.style.background = 'rgba(59,130,246,0.85)';
        cursor.style.boxShadow  = '0 0 0 3px rgba(59,130,246,0.3)';
      }
    }, 220);

    // Find the element under the cursor (temporarily hide the cursor dot first
    // so elementFromPoint doesn't return the cursor itself).
    cursor.style.display = 'none';
    const el = document.elementFromPoint(cursorX, cursorY);
    cursor.style.display = 'block';

    if (el) {
      el.dispatchEvent(new MouseEvent('click', {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: cursorX,
        clientY: cursorY,
      }));
    }
  }

  // ── Message listener ──────────────────────────────────────────────────────────
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type !== 'PAGE_ACTION') return;

    switch (msg.action) {
      case 'SCROLL_UP':       window.scrollBy({ top: -220, behavior: 'smooth' }); break;
      case 'SCROLL_DOWN':     window.scrollBy({ top:  220, behavior: 'smooth' }); break;
      case 'SCROLL_UP_FAST':  window.scrollBy({ top: -550, behavior: 'smooth' }); break;
      case 'SCROLL_DOWN_FAST':window.scrollBy({ top:  550, behavior: 'smooth' }); break;

      case 'BACK':            history.back();    break;
      case 'FORWARD':         history.forward(); break;

      case 'MOVE_CURSOR':     moveCursor(msg.nx, msg.ny); break;
      case 'CLICK':           simulateClick();            break;
      case 'HIDE_CURSOR':     hideCursor();               break;

      case 'DISABLE':
        hideCursor();
        break;
    }
  });
})();
