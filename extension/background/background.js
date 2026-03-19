// ─── Open side panel when the extension icon is clicked ───────────────────────
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

// ─── Message router ────────────────────────────────────────────────────────────
// Receives gesture action requests from the side panel and either:
//   a) Handles tab-level actions here (next tab, close tab, reload, new tab), or
//   b) Forwards page-level actions to the content script in the active tab.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'TAB_ACTION') {
    handleTabAction(message.action);
    sendResponse({ ok: true });
    return;
  }

  if (message.type === 'PAGE_ACTION') {
    forwardToContentScript(message);
    sendResponse({ ok: true });
    return;
  }

  // Side panel asks us to open a focused popup window so Chrome surfaces
  // the camera permission dialog (it won't appear inside a side panel).
  if (message.type === 'OPEN_PERMISSION_POPUP') {
    chrome.windows.create({
      url:    chrome.runtime.getURL('permission/permission.html'),
      type:   'popup',
      width:  420,
      height: 180,
      focused: true,
    });
    sendResponse({ ok: true });
    return;
  }
});

// ─── Tab-level actions ─────────────────────────────────────────────────────────
function handleTabAction(action) {
  switch (action) {

    case 'NEXT_TAB':
      chrome.tabs.query({ currentWindow: true }, (tabs) => {
        chrome.tabs.query({ active: true, currentWindow: true }, (activeTabs) => {
          if (!activeTabs.length) return;
          const nextIndex = (activeTabs[0].index + 1) % tabs.length;
          chrome.tabs.update(tabs[nextIndex].id, { active: true });
        });
      });
      break;

    case 'PREV_TAB':
      chrome.tabs.query({ currentWindow: true }, (tabs) => {
        chrome.tabs.query({ active: true, currentWindow: true }, (activeTabs) => {
          if (!activeTabs.length) return;
          const prevIndex = (activeTabs[0].index - 1 + tabs.length) % tabs.length;
          chrome.tabs.update(tabs[prevIndex].id, { active: true });
        });
      });
      break;

    case 'RELOAD':
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length) chrome.tabs.reload(tabs[0].id);
      });
      break;

    case 'NEW_TAB':
      chrome.tabs.create({});
      break;
  }
}

// ─── Forward page-level messages to active tab's content script ────────────────
// Programmatically injects content.js if not already present (handles tabs that
// were already open when the extension was loaded or reloaded). The IIFE guard
// in content.js prevents any double-execution side-effects.
async function forwardToContentScript(message) {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tabs.length) return;
  const tabId = tabs[0].id;

  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ['content/content.js'],
    });
  } catch (_) {
    // Tab is a privileged page (chrome://, extensions store, etc.) — skip.
    return;
  }

  chrome.tabs.sendMessage(tabId, message).catch(() => {});
}
