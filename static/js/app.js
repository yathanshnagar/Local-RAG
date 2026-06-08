// App State
let currentSessionId = null;
let selectedDocIds = [];
let pollingInterval = null;
let currentCitations = {}; // Maps index -> citation details

// DOM Elements
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const chatMessages = document.getElementById("chat-messages");
const chatWelcome = document.getElementById("chat-welcome");
const activeSessionTitle = document.getElementById("active-session-title");
const chatSubInfo = document.getElementById("chat-sub-info");
const typingStatus = document.getElementById("typing-status");

const chatModelSelect = document.getElementById("chat-model-select");
const embedModelSelect = document.getElementById("embed-model-select");
const ollamaStatusDot = document.getElementById("ollama-status-dot");
const ollamaStatusText = document.getElementById("ollama-status-text");

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");

const sessionList = document.getElementById("session-list");
const newChatBtn = document.getElementById("new-chat-btn");

const sourcePreview = document.getElementById("source-preview");
const previewMeta = document.getElementById("preview-meta");
const previewContent = document.getElementById("preview-content");
const closePreviewBtn = document.getElementById("close-preview-btn");

const pullModal = document.getElementById("pull-modal");
const openPullModalBtn = document.getElementById("open-pull-modal-btn");
const closePullModal = document.getElementById("close-pull-modal");
const pullModelInput = document.getElementById("pull-model-input");
const pullModelBtn = document.getElementById("pull-model-btn");
const pullProgressSection = document.getElementById("pull-progress-section");
const pullStatusText = document.getElementById("pull-status-text");
const pullProgressBar = document.getElementById("pull-progress-bar");
const pullLog = document.getElementById("pull-log");

const ragIndicator = document.getElementById("rag-indicator");

// Initialize Lucide Icons
function updateIcons() {
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// App Initialization
document.addEventListener("DOMContentLoaded", () => {
    updateIcons();
    checkOllamaStatus();
    loadSessions();
    loadDocuments();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    // Send Message
    sendBtn.addEventListener("click", sendMessage);
    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    chatInput.addEventListener("input", () => {
        chatInput.style.height = "auto";
        chatInput.style.height = (chatInput.scrollHeight) + "px";
    });

    // New Session
    newChatBtn.addEventListener("click", createNewSession);

    // Dropzone logic
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", handleFileSelect);
    
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            uploadFiles(e.dataTransfer.files);
        }
    });

    // Pull Model Modal
    openPullModalBtn.addEventListener("click", () => {
        pullModal.classList.add("active");
        pullModelInput.value = "";
        pullProgressSection.style.display = "none";
        pullLog.innerHTML = "";
    });
    closePullModal.addEventListener("click", () => pullModal.classList.remove("active"));
    pullModelBtn.addEventListener("click", handlePullModel);

    // Source Preview Close
    closePreviewBtn.addEventListener("click", () => {
        sourcePreview.style.display = "none";
    });
}

// Ollama API Actions
async function checkOllamaStatus() {
    try {
        const response = await fetch("/ollama/models");
        const data = await response.json();
        
        if (data.status === "connected") {
            ollamaStatusDot.className = "dot online";
            ollamaStatusText.textContent = "Online";
            
            // Populate chat model select dropdown
            chatModelSelect.innerHTML = "";
            if (data.models.length > 0) {
                data.models.forEach(model => {
                    // Try to filter out embedding models from LLM list (pure helper convenience)
                    if (!model.includes("embed")) {
                        const opt = document.createElement("option");
                        opt.value = model;
                        opt.textContent = model;
                        chatModelSelect.appendChild(opt);
                    }
                });
                
                // If we ended up with nothing because all have 'embed'
                if (chatModelSelect.children.length === 0) {
                    data.models.forEach(model => {
                        const opt = document.createElement("option");
                        opt.value = model;
                        opt.textContent = model;
                        chatModelSelect.appendChild(opt);
                    });
                }
            } else {
                chatModelSelect.innerHTML = `<option value="">No models. Pull one!</option>`;
            }

            // Populate embedding model select dropdown
            embedModelSelect.innerHTML = "";
            const defaultEmbedModels = ["nomic-embed-text", "all-minilm"];
            const installedModels = data.models || [];
            let addedInstalled = false;

            // Add any installed models that look like embedding models
            installedModels.forEach(model => {
                if (model.includes("embed") || model.includes("minilm") || model.includes("mxbai")) {
                    const opt = document.createElement("option");
                    opt.value = model;
                    opt.textContent = model;
                    embedModelSelect.appendChild(opt);
                    addedInstalled = true;
                }
            });

            // Add defaults, marking them as not downloaded if not installed
            defaultEmbedModels.forEach(model => {
                const isInstalled = installedModels.some(m => m.startsWith(model) || model.startsWith(m));
                if (!isInstalled) {
                    const opt = document.createElement("option");
                    opt.value = model;
                    opt.textContent = `${model} (Not downloaded - pull first)`;
                    embedModelSelect.appendChild(opt);
                }
            });
        } else {
            ollamaStatusDot.className = "dot offline";
            ollamaStatusText.textContent = "Offline";
            chatModelSelect.innerHTML = `<option value="">Ollama offline</option>`;
            embedModelSelect.innerHTML = `<option value="nomic-embed-text">nomic-embed-text</option>`;
        }
    } catch (e) {
        ollamaStatusDot.className = "dot offline";
        ollamaStatusText.textContent = "Offline";
        chatModelSelect.innerHTML = `<option value="">Connection error</option>`;
        embedModelSelect.innerHTML = `<option value="nomic-embed-text">nomic-embed-text</option>`;
    }
}

async function handlePullModel() {
    const modelName = pullModelInput.value.trim();
    if (!modelName) return;

    pullProgressSection.style.display = "block";
    pullStatusText.textContent = "Starting download...";
    pullProgressBar.style.width = "0%";
    pullLog.innerHTML = `<div>Sending pull request to Ollama for: ${modelName}...</div>`;
    pullModelBtn.disabled = true;

    try {
        const response = await fetch("/ollama/pull", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: modelName })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const progress = JSON.parse(line);
                        
                        if (progress.error) {
                            pullLog.innerHTML += `<div style="color: var(--danger)">Error: ${progress.error}</div>`;
                            pullStatusText.textContent = "Download failed";
                            pullModelBtn.disabled = false;
                            return;
                        }

                        if (progress.status) {
                            let logMsg = progress.status;
                            if (progress.completed && progress.total) {
                                const percent = Math.round((progress.completed / progress.total) * 100);
                                pullProgressBar.style.width = `${percent}%`;
                                logMsg += ` - ${percent}% (${Math.round(progress.completed / 1024 / 1024)}MB / ${Math.round(progress.total / 1024 / 1024)}MB)`;
                                pullStatusText.textContent = `Downloading: ${percent}%`;
                            } else {
                                pullStatusText.textContent = progress.status;
                            }
                            
                            pullLog.innerHTML += `<div>${logMsg}</div>`;
                            pullLog.scrollTop = pullLog.scrollHeight;
                        }
                    } catch (e) {
                        // Not JSON, just display raw log line
                        pullLog.innerHTML += `<div>${line}</div>`;
                    }
                }
            }
        }
        
        pullStatusText.textContent = "Completed successfully!";
        pullProgressBar.style.width = "100%";
        pullModelBtn.disabled = false;
        
        // Refresh models list
        setTimeout(() => {
            checkOllamaStatus();
            pullModal.classList.remove("active");
        }, 1500);
        
    } catch (err) {
        pullLog.innerHTML += `<div style="color: var(--danger)">Connection Error: ${err.message}</div>`;
        pullStatusText.textContent = "Network error";
        pullModelBtn.disabled = false;
    }
}

// Ingestion Actions
function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        uploadFiles(e.target.files);
    }
}

async function uploadFiles(fileList) {
    const embeddingModel = embedModelSelect.value;
    
    for (const file of fileList) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("embedding_model", embeddingModel);
        
        // Render upload progress block in sidebar
        const fileItem = appendUploadItem(file.name);
        
        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });
            const result = await response.json();
            
            if (response.ok) {
                fileItem.querySelector(".list-item-status").className = "list-item-status status-queued";
                fileItem.querySelector(".list-item-status").textContent = "queued";
                // Start polling database status
                startDocumentsPolling();
            } else {
                updateFileItemFailed(fileItem, result.detail || "Upload error");
            }
        } catch (e) {
            updateFileItemFailed(fileItem, e.message);
        }
    }
}

function appendUploadItem(name) {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
        <div class="list-item-info">
            <span class="list-item-name" title="${name}">${name}</span>
            <span class="list-item-meta">
                <span class="list-item-status status-processing">uploading</span>
                <span class="spinner"></span>
            </span>
        </div>
    `;
    fileList.insertBefore(item, fileList.firstChild);
    return item;
}

function updateFileItemFailed(element, errorText) {
    const statusSpan = element.querySelector(".list-item-status");
    statusSpan.className = "list-item-status status-failed";
    statusSpan.textContent = "failed";
    
    const spinner = element.querySelector(".spinner");
    if (spinner) spinner.remove();
    
    // Add title warning
    element.title = errorText;
}

async function loadDocuments() {
    try {
        const response = await fetch("/documents");
        const docs = await response.json();
        
        fileList.innerHTML = "";
        let hasUnfinished = false;
        
        if (docs.length === 0) {
            fileList.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 1rem;">No files uploaded yet</div>`;
            return;
        }

        docs.forEach(doc => {
            const isCompleted = doc.status === "completed";
            const isFailed = doc.status === "failed";
            
            if (!isCompleted && !isFailed) {
                hasUnfinished = true;
            }
            
            // Check if document is checked for active RAG selection
            const isChecked = selectedDocIds.includes(doc.id);
            
            const item = document.createElement("div");
            item.className = "list-item";
            item.innerHTML = `
                <div style="display: flex; align-items: center; gap: 0.5rem; width: 85%;">
                    ${isCompleted ? `<input type="checkbox" class="doc-select-checkbox" data-id="${doc.id}" ${isChecked ? "checked" : ""}>` : ""}
                    <div class="list-item-info" style="width: 100%;">
                        <span class="list-item-name" title="${doc.filename}">${doc.filename}</span>
                        <span class="list-item-meta">
                            <span class="list-item-status status-${doc.status}">${doc.status}</span>
                            ${(!isCompleted && !isFailed) ? `<span class="spinner"></span>` : ""}
                        </span>
                    </div>
                </div>
                <button class="action-btn delete-doc-btn" data-id="${doc.id}" title="Delete document">
                    <i data-lucide="trash-2" style="width: 14px; height: 14px;"></i>
                </button>
            `;
            
            // Delete action
            item.querySelector(".delete-doc-btn").addEventListener("click", () => deleteDoc(doc.id));
            
            // Checkbox change action
            const checkbox = item.querySelector(".doc-select-checkbox");
            if (checkbox) {
                checkbox.addEventListener("change", (e) => {
                    const id = parseInt(e.target.dataset.id);
                    if (e.target.checked) {
                        if (!selectedDocIds.includes(id)) selectedDocIds.push(id);
                    } else {
                        selectedDocIds = selectedDocIds.filter(x => x !== id);
                    }
                    updateRagIndicator();
                });
            }
            
            if (isFailed && doc.error_message) {
                item.title = `Error: ${doc.error_message}`;
            }

            fileList.appendChild(item);
        });
        
        updateIcons();
        updateRagIndicator();
        
        if (hasUnfinished) {
            startDocumentsPolling();
        } else {
            stopDocumentsPolling();
        }
    } catch (e) {
        console.error("Failed to load documents: ", e);
    }
}

function startDocumentsPolling() {
    if (!pollingInterval) {
        pollingInterval = setInterval(loadDocuments, 3000);
    }
}

function stopDocumentsPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function deleteDoc(docId) {
    if (confirm("Are you sure you want to delete this document? All associated vectors will be erased.")) {
        try {
            await fetch(`/documents/${docId}`, { method: "DELETE" });
            selectedDocIds = selectedDocIds.filter(id => id !== docId);
            loadDocuments();
        } catch (e) {
            alert("Delete failed: " + e.message);
        }
    }
}

function updateRagIndicator() {
    if (selectedDocIds.length > 0) {
        ragIndicator.style.display = "flex";
        chatSubInfo.textContent = `RAG searching over ${selectedDocIds.length} selected file(s)`;
    } else {
        ragIndicator.style.display = "none";
        chatSubInfo.textContent = "Direct conversation mode (no context documents selected)";
    }
}

// Session Actions
async function loadSessions() {
    try {
        const response = await fetch("/sessions");
        const sessions = await response.json();
        
        sessionList.innerHTML = "";
        
        if (sessions.length === 0) {
            // Auto create first session if none exists
            createNewSession();
            return;
        }

        sessions.forEach(session => {
            const isActive = currentSessionId === session.id;
            const item = document.createElement("div");
            item.className = `session-item ${isActive ? "active" : ""}`;
            item.dataset.id = session.id;
            
            // Simple display title formatting
            const titleText = session.title || `Chat Session ${session.id}`;
            item.innerHTML = `
                <span class="session-title" title="${titleText}">${titleText}</span>
                <button class="action-btn delete-session-btn" data-id="${session.id}">
                    <i data-lucide="trash" style="width: 12px; height: 12px;"></i>
                </button>
            `;
            
            item.addEventListener("click", (e) => {
                // Ignore if clicking the delete button
                if (e.target.closest(".delete-session-btn")) return;
                selectSession(session.id, titleText);
            });
            
            item.querySelector(".delete-session-btn").addEventListener("click", () => deleteSession(session.id));
            
            sessionList.appendChild(item);
        });
        
        updateIcons();
        
        // If current session doesn't exist, select the first one
        if (!currentSessionId && sessions.length > 0) {
            selectSession(sessions[0].id, sessions[0].title);
        }
    } catch (e) {
        console.error("Error loading chat sessions:", e);
    }
}

async function createNewSession() {
    try {
        const response = await fetch("/sessions", { method: "POST" });
        const result = await response.json();
        currentSessionId = result.session_id;
        
        loadSessions();
    } catch (e) {
        alert("Failed to create new session: " + e.message);
    }
}

async function selectSession(sessionId, title) {
    currentSessionId = sessionId;
    activeSessionTitle.textContent = title || `Chat Session ${sessionId}`;
    
    // Highlight active session
    document.querySelectorAll(".session-item").forEach(item => {
        if (parseInt(item.dataset.id) === sessionId) {
            item.classList.add("active");
        } else {
            item.classList.remove("active");
        }
    });
    
    // Load messages
    try {
        const response = await fetch(`/sessions/${sessionId}/messages`);
        const messages = await response.json();
        
        chatMessages.innerHTML = "";
        currentCitations = {};
        
        if (messages.length === 0) {
            chatMessages.appendChild(chatWelcome);
            chatWelcome.style.display = "flex";
            chatInput.disabled = false;
            sendBtn.disabled = false;
            return;
        }
        
        chatWelcome.style.display = "none";
        
        messages.forEach(msg => {
            renderMessage(msg.role, msg.content, msg.citations);
        });
        
        chatInput.disabled = false;
        sendBtn.disabled = false;
        scrollToBottom();
    } catch (e) {
        console.error("Error loading session messages:", e);
    }
}

async function deleteSession(sessionId) {
    if (confirm("Delete this conversation?")) {
        try {
            await fetch(`/sessions/${sessionId}`, { method: "DELETE" });
            if (currentSessionId === sessionId) {
                currentSessionId = null;
            }
            loadSessions();
        } catch (e) {
            alert("Failed to delete session: " + e.message);
        }
    }
}

// Chat Send & Stream Logic
async function sendMessage() {
    const messageText = chatInput.value.trim();
    if (!messageText || !currentSessionId) return;

    // Clear input
    chatInput.value = "";
    chatInput.style.height = "auto";
    
    // Disable inputs during streaming
    chatInput.disabled = true;
    sendBtn.disabled = true;
    chatWelcome.style.display = "none";
    
    // Render user message instantly
    renderMessage("user", messageText);
    scrollToBottom();
    
    // Retrieve settings
    const chatModel = chatModelSelect.value;
    const embeddingModel = embedModelSelect.value;
    
    if (!chatModel) {
        renderMessage("assistant", "Error: No chat LLM selected or Ollama is offline. Please select a model first.", null, true);
        chatInput.disabled = false;
        sendBtn.disabled = false;
        return;
    }
    
    // Create assistant message bubble block placeholder
    const assistantBubble = renderMessagePlaceholder();
    scrollToBottom();
    
    typingStatus.textContent = "AI is thinking...";
    
    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: messageText,
                session_id: currentSessionId,
                chat_model: chatModel,
                embedding_model: embeddingModel,
                doc_ids: selectedDocIds.length > 0 ? selectedDocIds : null
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            updateMessagePlaceholderError(assistantBubble, errData.detail || "Server error occurred");
            resetInputState();
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let assistantResponseText = "";
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep half-parsed line

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const parsed = JSON.parse(line);
                        
                        if (parsed.error) {
                            updateMessagePlaceholderError(assistantBubble, parsed.error);
                            resetInputState();
                            return;
                        }
                        
                        if (parsed.warning) {
                            console.warn(parsed.warning);
                        }
                        
                        if (parsed.token) {
                            assistantResponseText += parsed.token;
                            updateMessagePlaceholderText(assistantBubble, assistantResponseText);
                            scrollToBottom();
                        }
                        
                        if (parsed.citations) {
                            renderCitations(assistantBubble, parsed.citations);
                            scrollToBottom();
                        }
                    } catch (e) {
                        // Keep parsing lines even if single lines crash
                        console.error("Parse error on stream chunk:", e);
                    }
                }
            }
        }
        
    } catch (e) {
        updateMessagePlaceholderError(assistantBubble, "Network connection failure: " + e.message);
    } finally {
        resetInputState();
    }
}

function resetInputState() {
    typingStatus.textContent = "";
    chatInput.disabled = false;
    sendBtn.disabled = false;
    chatInput.focus();
}

// Markdown and Citation Helpers
function formatMarkdown(text) {
    if (!text) return "";
    let html = text;
    
    // Prevent XSS Injection
    html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // Fenced Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function(match, lang, code) {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    
    // Lists parsing
    const lines = html.split('\n');
    let inList = false;
    let listHtml = [];
    
    lines.forEach(line => {
        const trimmed = line.trim();
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('+ ')) {
            if (!inList) {
                listHtml.push('<ul>');
                inList = true;
            }
            listHtml.push(`<li>${trimmed.substring(2)}</li>`);
        } else {
            if (inList) {
                listHtml.push('</ul>');
                inList = false;
            }
            listHtml.push(line);
        }
    });
    
    if (inList) listHtml.push('</ul>');
    html = listHtml.join('\n');
    
    // Convert newlines to breaks (ignoring pre tags)
    html = html.split(/<pre[\s\S]*?<\/pre>/g).map((part, index, arr) => {
        if (index % 2 === 0) {
            return part.replace(/\n/g, '<br>');
        }
        return part; // Re-evaluate code segment without newline replacement
    });
    
    // Reassemble parts if we had pre tags (basic markdown pre splitter logic)
    // For simplicity, we just use a replacement of double breaks
    html = html.join('<pre>...</pre>'); // Standard fallback, let's keep it simple
    
    // Better simple newline replacement
    // Simple table parser
    html = parseTables(text);
    
    // Highlight citations like [1], [2] as clickable indexes
    html = html.replace(/\[([0-9]+)\]/g, '<span class="citation-index" onclick="previewCitationFromText($1)">$1</span>');
    
    return html;
}

function parseTables(text) {
    const lines = text.split('\n');
    let tableHtml = [];
    let inTable = false;
    
    lines.forEach(line => {
        const trimmed = line.trim();
        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
            const cells = trimmed.split('|').slice(1, -1).map(c => c.trim());
            if (!inTable) {
                tableHtml.push('<table><thead><tr>');
                cells.forEach(c => tableHtml.push(`<th>${c}</th>`));
                tableHtml.push('</tr></thead><tbody>');
                inTable = true;
            } else if (trimmed.includes('---')) {
                // Separator row, skip
            } else {
                tableHtml.push('<tr>');
                cells.forEach(c => tableHtml.push(`<td>${c}</td>`));
                tableHtml.push('</tr>');
            }
        } else {
            if (inTable) {
                tableHtml.push('</tbody></table>');
                inTable = false;
            }
            // Parse other markdown logic
            let lineText = trimmed
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                .replace(/`([^`]+)`/g, '<code>$1</code>');
            tableHtml.push(lineText);
        }
    });
    
    if (inTable) tableHtml.push('</tbody></table>');
    
    // Join lines and replace newlines with brs, ignoring tables
    let processed = tableHtml.join('\n');
    processed = processed.replace(/\n/g, '<br>');
    // Clean up br tags generated inside table structure
    processed = processed.replace(/<\/tr><br>/g, '</tr>').replace(/<\/thead><br>/g, '</thead>').replace(/<\/table><br>/g, '</table>');
    return processed;
}

// Rendering message structures
function renderMessage(role, content, citations = null, isError = false) {
    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper";
    
    const messageEl = document.createElement("div");
    messageEl.className = `message ${role}`;
    
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = role === "user" ? "You" : "Local Assistant";
    
    const body = document.createElement("div");
    body.className = "message-body";
    
    if (isError) {
        body.style.color = "var(--danger)";
        body.textContent = content;
    } else {
        body.innerHTML = formatMarkdown(content);
    }
    
    messageEl.appendChild(meta);
    messageEl.appendChild(body);
    
    if (citations && citations.length > 0) {
        renderCitations(messageEl, citations);
    }
    
    wrapper.appendChild(messageEl);
    chatMessages.appendChild(wrapper);
    return wrapper;
}

function renderMessagePlaceholder() {
    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper";
    
    const messageEl = document.createElement("div");
    messageEl.className = "message assistant";
    
    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = "Local Assistant";
    
    const body = document.createElement("div");
    body.className = "message-body";
    body.innerHTML = `<span style="font-style: italic; color: var(--text-secondary)">Thinking...</span>`;
    
    messageEl.appendChild(meta);
    messageEl.appendChild(body);
    
    wrapper.appendChild(messageEl);
    chatMessages.appendChild(wrapper);
    return messageEl;
}

function updateMessagePlaceholderText(element, text) {
    const body = element.querySelector(".message-body");
    body.innerHTML = formatMarkdown(text);
}

function updateMessagePlaceholderError(element, err) {
    const body = element.querySelector(".message-body");
    body.style.color = "var(--danger)";
    body.textContent = "Error: " + err;
}

function renderCitations(messageElement, citations) {
    const citationsBox = document.createElement("div");
    citationsBox.className = "citations-box";
    
    citations.forEach(cit => {
        // Save citation globally for clickable index reference
        currentCitations[cit.index] = cit;
        
        const chip = document.createElement("button");
        chip.className = "citation-chip";
        chip.innerHTML = `
            <span class="citation-index">${cit.index}</span>
            <span>${cit.filename}</span>
        `;
        
        chip.addEventListener("click", () => {
            previewCitation(cit);
        });
        
        citationsBox.appendChild(chip);
    });
    
    messageElement.appendChild(citationsBox);
}

function previewCitation(cit) {
    previewMeta.innerHTML = `<strong>File:</strong> ${cit.filename} ${cit.page_num ? `| <strong>Page:</strong> ${cit.page_num}` : ""}`;
    previewContent.textContent = cit.text;
    sourcePreview.style.display = "flex";
}

// Clickable index citation in text
window.previewCitationFromText = function(index) {
    const cit = currentCitations[index];
    if (cit) {
        previewCitation(cit);
    }
};

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
