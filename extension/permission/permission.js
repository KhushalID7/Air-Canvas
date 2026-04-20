navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    stream.getTracks().forEach((t) => t.stop());

    chrome.runtime.sendMessage({ type: 'CAMERA_PERMISSION', granted: true });

    window.close();
  })
  .catch((err) => {
    document.getElementById('err').textContent = err.name + ': ' + err.message;

    chrome.runtime.sendMessage({ type: 'CAMERA_PERMISSION', granted: false, error: err.message });
  });
