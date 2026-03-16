// This page runs in a focused popup window — Chrome reliably shows the
// camera permission dialog here, unlike inside a side panel.

navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    // Permission granted. Release the stream immediately — the side panel's
    // MediaPipe Camera will open its own stream after receiving this message.
    stream.getTracks().forEach((t) => t.stop());

    // Broadcast to all extension contexts (side panel will receive this).
    chrome.runtime.sendMessage({ type: 'CAMERA_PERMISSION', granted: true });

    // Close this popup window.
    window.close();
  })
  .catch((err) => {
    // Show the error inside the popup so the user knows what happened.
    document.getElementById('err').textContent = err.name + ': ' + err.message;

    // Notify the side panel that permission was denied.
    chrome.runtime.sendMessage({ type: 'CAMERA_PERMISSION', granted: false, error: err.message });
  });
