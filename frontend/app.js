// ============================================================
// RAG Chat Assistant - Frontend JavaScript
// ============================================================
// This script handles:
// - Session management (localStorage)
// - Sending messages to the backend API
// - Displaying messages with markdown rendering
// - Loading indicators, timestamps
// - Suggestion pills and new chat
// ============================================================

// ---- Session Management ----
// Each browser session gets a unique ID stored in localStorage.
// This ID is sent with every message so the backend can track history.

function generateSessionId() {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Load existing session ID or create a new one
let sessionId = localStorage.getItem('rag_session_id');
if (!sessionId) {
  sessionId = generateSessionId();
  localStorage.setItem('rag_session_id', sessionId);
}

// Display the session ID in the sidebar
document.getElementById('sessionDisplay').textContent = sessionId.slice(0, 20) + '...';

// ---- State ----
let isLoading = false;    // Prevent double-sends
let messageCount = 0;     // Track message count

// ---- DOM References ----
const messagesContainer = document.getElementById('messagesContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const welcomeScreen = document.getElementById('welcomeScreen');
const sidebar = document.getElementById('sidebar');

// ============================================================
// MAIN FUNCTION: Send a message
// ============================================================
async function sendMessage() {
  const message = messageInput.value.trim();

  // Don't send empty messages or if already loading
  if (!message || isLoading) return;

  // Hide the welcome screen once first message is sent
  if (welcomeScreen) {
    welcomeScreen.style.display = 'none';
  }

  // Clear the input field and reset height
  messageInput.value = '';
  messageInput.style.height = 'auto';

  // Add user message to the chat
  addMessage('user', message);

  // Show loading indicator while waiting for backend
  const loadingEl = showLoading();

  // Disable input during loading
  setLoading(true);

  try {
    // ---- Call the Backend API ----
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: sessionId,
        message: message
      })
    });

    // Parse the JSON response
    const data = await response.json();

    // Remove the loading indicator
    loadingEl.remove();

    if (!response.ok) {
      // Handle error responses from backend
      const errorMsg = data.detail?.error || data.error || 'An error occurred. Please try again.';
      addMessage('assistant', `⚠️ ${errorMsg}`, null, null, true);
    } else {
      // Display the assistant's reply
      addMessage('assistant', data.reply, data.tokensUsed, data.retrievedChunks);
    }

  } catch (error) {
    // Network error or server is down
    loadingEl.remove();
    addMessage(
      'assistant',
      '⚠️ Could not connect to the server. Please make sure the backend is running on http://localhost:8000',
      null, null, true
    );
  } finally {
    setLoading(false);
    scrollToBottom();
  }
}

// ============================================================
// Add a message bubble to the chat
// ============================================================
function addMessage(role, content, tokensUsed = null, retrievedChunks = null, isError = false) {
  messageCount++;

  const messageEl = document.createElement('div');
  messageEl.className = `message ${role}`;
  messageEl.id = `msg_${messageCount}`;

  const avatar = role === 'user' ? '👤' : '🧠';
  const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  // Render markdown for assistant messages
  let renderedContent = content;
  if (role === 'assistant' && typeof marked !== 'undefined') {
    try {
      renderedContent = marked.parse(content);
    } catch (e) {
      renderedContent = content; // Fallback to plain text
    }
  } else if (role === 'user') {
    // For user messages, escape HTML to prevent XSS
    renderedContent = escapeHtml(content);
  }

  // Build metadata tags for assistant messages
  let metaTags = `<span class="meta-tag">${timestamp}</span>`;
  if (role === 'assistant') {
    if (retrievedChunks !== null && retrievedChunks >= 0) {
      metaTags += `<span class="meta-tag chunks">📄 ${retrievedChunks} chunk${retrievedChunks !== 1 ? 's' : ''} retrieved</span>`;
    }
    if (tokensUsed) {
      metaTags += `<span class="meta-tag tokens">🪙 ${tokensUsed} tokens</span>`;
    }
  }

  messageEl.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-bubble ${isError ? 'error-bubble' : ''}">${renderedContent}</div>
      <div class="message-meta">${metaTags}</div>
    </div>
  `;

  messagesContainer.appendChild(messageEl);
  scrollToBottom();
}

// ============================================================
// Show thinking/loading indicator
// ============================================================
function showLoading() {
  const loadingEl = document.createElement('div');
  loadingEl.className = 'loading-message';
  loadingEl.id = 'loadingIndicator';

  loadingEl.innerHTML = `
    <div class="message-avatar" style="background: var(--bg-card); border: 1px solid var(--border);">🧠</div>
    <div class="thinking-bubble">
      <div class="thinking-dots">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <span>Searching knowledge base...</span>
    </div>
  `;

  messagesContainer.appendChild(loadingEl);
  scrollToBottom();
  return loadingEl;
}

// ============================================================
// UI Helpers
// ============================================================

// Enable/disable input during API calls
function setLoading(loading) {
  isLoading = loading;
  sendBtn.disabled = loading;
  messageInput.disabled = loading;

  if (loading) {
    messageInput.placeholder = 'Searching knowledge base and generating answer...';
  } else {
    messageInput.placeholder = 'Ask me anything... (e.g. How do I reset my password?)';
    messageInput.focus();
  }
}

// Scroll chat to the bottom
function scrollToBottom() {
  setTimeout(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }, 50);
}

// Auto-resize textarea as user types
function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// Handle Enter key (send) vs Shift+Enter (new line)
function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();  // Prevent actual newline
    sendMessage();
  }
}

// Toggle sidebar visibility
function toggleSidebar() {
  sidebar.classList.toggle('collapsed');
  sidebar.classList.toggle('mobile-open');
}

// Escape HTML to prevent XSS in user messages
function escapeHtml(text) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(text));
  return div.innerHTML;
}

// ============================================================
// Suggestion Pills
// ============================================================
function fillSuggestion(text) {
  messageInput.value = text;
  messageInput.focus();
  autoResize(messageInput);
}

// ============================================================
// New Chat
// ============================================================
async function startNewChat() {
  // Tell backend to clear this session's history
  try {
    await fetch(`/api/chat/clear?session_id=${sessionId}`, { method: 'POST' });
  } catch (e) {
    // Ignore errors — it's just cleanup
  }

  // Generate a new session ID
  sessionId = generateSessionId();
  localStorage.setItem('rag_session_id', sessionId);
  document.getElementById('sessionDisplay').textContent = sessionId.slice(0, 20) + '...';

  // Clear the chat display
  const messages = messagesContainer.querySelectorAll('.message, .loading-message');
  messages.forEach(el => el.remove());

  // Show welcome screen again
  if (welcomeScreen) {
    welcomeScreen.style.display = '';
  }

  messageCount = 0;
  messageInput.value = '';
  messageInput.focus();
}

// ============================================================
// File Upload Handling
// ============================================================
async function handleFileUpload(event) {
  const fileInput = event.target;
  const file = fileInput.files[0];
  if (!file) return;

  const allowedExtensions = ['.pdf', '.docx'];
  const fileName = file.name;
  const extension = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();

  if (!allowedExtensions.includes(extension)) {
    showUploadStatus(`Only PDF and DOCX files are allowed.`, 'error');
    fileInput.value = '';
    return;
  }

  const uploadBtnLabel = document.getElementById('uploadBtnLabel');
  const uploadBtnText = document.getElementById('uploadBtnText');
  const uploadProgress = document.getElementById('uploadProgress');
  const progressBarFill = document.getElementById('progressBarFill');
  const uploadStatus = document.getElementById('uploadStatus');

  // Update UI to uploading state
  uploadBtnLabel.style.pointerEvents = 'none';
  uploadBtnText.textContent = 'Ingesting document...';
  uploadProgress.style.display = 'block';
  progressBarFill.style.width = '0%';
  showUploadStatus('Ingesting document into ChromaDB...', 'info');

  // Create FormData
  const formData = new FormData();
  formData.append('file', file);

  // Upload using XMLHttpRequest to track progress
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload', true);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const percentage = Math.round((e.loaded / e.total) * 100);
      // We reserve up to 80% for upload, and the rest for backend ingestion
      progressBarFill.style.width = (percentage * 0.8) + '%';
    }
  };

  xhr.onload = () => {
    // Reset button pointer events
    uploadBtnLabel.style.pointerEvents = '';
    fileInput.value = ''; // Reset file input

    if (xhr.status >= 200 && xhr.status < 300) {
      try {
        const response = JSON.parse(xhr.responseText);
        progressBarFill.style.width = '100%';
        showUploadStatus(`Indexed! Added ${response.chunks_added} chunks.`, 'success');
        
        // Add a notification in the chat
        if (welcomeScreen) {
          welcomeScreen.style.display = 'none';
        }
        addMessage('assistant', `✅ **Document Uploaded & Indexed**\n\nI have successfully indexed \`${fileName}\` (${response.chunks_added} chunks) into ChromaDB. You can now ask questions about its content!`, null, null);
        
        // Hide progress bar after 3 seconds
        setTimeout(() => {
          uploadProgress.style.display = 'none';
          progressBarFill.style.width = '0%';
        }, 3000);
      } catch (err) {
        showUploadStatus('Ingestion succeeded, but response parsing failed.', 'error');
        uploadProgress.style.display = 'none';
      }
    } else {
      let errorMsg = 'Failed to index document.';
      try {
        const response = JSON.parse(xhr.responseText);
        errorMsg = response.detail?.error || response.error || errorMsg;
      } catch (err) {}
      showUploadStatus(errorMsg, 'error');
      uploadProgress.style.display = 'none';
    }
  };

  xhr.onerror = () => {
    uploadBtnLabel.style.pointerEvents = '';
    fileInput.value = '';
    showUploadStatus('Network error during upload.', 'error');
    uploadProgress.style.display = 'none';
  };

  xhr.send(formData);
}

function showUploadStatus(message, type) {
  const statusDiv = document.getElementById('uploadStatus');
  statusDiv.textContent = message;
  statusDiv.className = 'upload-status ' + type;
}

// ============================================================
// On page load
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  messageInput.focus();
  console.log('🧠 RAG Chat Assistant loaded. Session:', sessionId);
});
