chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

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
    return;
  }

  chrome.tabs.sendMessage(tabId, message).catch(() => {});
}
