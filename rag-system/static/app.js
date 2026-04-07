/**
 * RAG System — Client-side JavaScript
 * Handles file uploads, chat interactions, and streaming responses.
 */

// ── DOM Elements ─────────────────────────────────────────────
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const docsList = document.getElementById('docsList');
const docsStats = document.getElementById('docsStats');
const statusDot = document.getElementById('statusDot');
const systemInfo = document.getElementById('systemInfo');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messages');
const chatContainer = document.getElementById('chatContainer');
const queryForm = document.getElementById('queryForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');

// ── State ────────────────────────────────────────────────────
let isProcessing = false;

// ── Initialization ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadDocuments();
    setupEventListeners();
});

function setupEventListeners() {
    // Sidebar toggle
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    // Upload zone
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFiles(fileInput.files);
            fileInput.value = '';
        }
    });

    // Query form
    queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        handleQuery();
    });

    // Textarea auto-resize + enter handling
    questionInput.addEventListener('input', () => {
        questionInput.style.height = 'auto';
        questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
        sendBtn.disabled = !questionInput.value.trim() || isProcessing;
    });

    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (questionInput.value.trim() && !isProcessing) {
                handleQuery();
            }
        }
    });

    // Hint cards
    document.querySelectorAll('.hint-card').forEach(card => {
        card.addEventListener('click', () => {
            const q = card.dataset.question;
            questionInput.value = q;
            questionInput.dispatchEvent(new Event('input'));
            handleQuery();
        });
    });
}

// ── Health Check ─────────────────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            const data = await res.json();
            statusDot.classList.add('online');
            // Add more system info
            systemInfo.innerHTML = `
                <div class="info-row"><span class="info-label">Status</span><span class="status-dot online"></span></div>
                <div class="info-row"><span class="info-label">Embedder</span><span class="info-value">${data.embedding_model}</span></div>
                <div class="info-row"><span class="info-label">LLM</span><span class="info-value">${data.llm_model}</span></div>
                <div class="info-row"><span class="info-label">Chunks</span><span class="info-value">${data.documents_indexed}</span></div>
            `;
        } else {
            statusDot.classList.remove('online');
        }
    } catch {
        statusDot.classList.remove('online');
    }
}

// ── Document Management ──────────────────────────────────────
async function loadDocuments() {
    try {
        const res = await fetch('/api/documents/');
        if (!res.ok) return;
        const data = await res.json();

        if (data.documents.length === 0) {
            docsList.innerHTML = '<div class="docs-empty">No documents yet</div>';
            docsStats.textContent = '';
            return;
        }

        docsList.innerHTML = data.documents.map(doc => `
            <div class="doc-item">
                <svg class="doc-item-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>
                    <path d="M14 2v4a2 2 0 0 0 2 2h4"/>
                </svg>
                <span class="doc-item-name">${doc.name}</span>
                <span class="doc-item-size">${formatBytes(doc.size_bytes)}</span>
            </div>
        `).join('');

        docsStats.textContent = `${data.total_chunks} chunks indexed across ${data.documents.length} document(s)`;
    } catch {
        // Silently fail
    }
}

async function handleFiles(files) {
    for (const file of files) {
        await uploadFile(file);
    }
    await loadDocuments();
    checkHealth();
}

async function uploadFile(file) {
    uploadProgress.style.display = 'block';
    progressFill.style.width = '20%';
    progressText.textContent = `Uploading ${file.name}…`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        progressFill.style.width = '50%';
        progressText.textContent = `Processing ${file.name}…`;

        const res = await fetch('/api/documents/', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await res.json();
        progressFill.style.width = '100%';
        progressText.textContent = `✓ ${data.message}`;

        setTimeout(() => {
            uploadProgress.style.display = 'none';
            progressFill.style.width = '0%';
        }, 3000);
    } catch (err) {
        progressFill.style.width = '100%';
        progressFill.style.background = '#ef4444';
        progressText.textContent = `✗ ${err.message}`;

        setTimeout(() => {
            uploadProgress.style.display = 'none';
            progressFill.style.width = '0%';
            progressFill.style.background = '';
        }, 4000);
    }
}

// ── Chat / Query ─────────────────────────────────────────────
async function handleQuery() {
    const question = questionInput.value.trim();
    if (!question || isProcessing) return;

    isProcessing = true;
    sendBtn.disabled = true;
    questionInput.value = '';
    questionInput.style.height = 'auto';

    // Hide welcome screen
    welcomeScreen.style.display = 'none';

    // Add user message
    addMessage('user', question);

    // Add assistant placeholder with typing indicator
    const assistantEl = addMessage('assistant', '', true);
    const contentEl = assistantEl.querySelector('.message-content');

    try {
        const res = await fetch('/api/query/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Query failed');
        }

        // Read SSE stream
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let answer = '';
        let sources = [];

        // Remove typing indicator
        const typingEl = contentEl.querySelector('.typing-indicator');
        if (typingEl) typingEl.remove();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const text = decoder.decode(value, { stream: true });
            const lines = text.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.slice(6);
                try {
                    const event = JSON.parse(jsonStr);
                    if (event.type === 'token') {
                        answer += event.content;
                        contentEl.innerHTML = formatMarkdown(answer);
                        scrollToBottom();
                    } else if (event.type === 'sources') {
                        sources = event.content;
                    } else if (event.type === 'error') {
                        throw new Error(event.content);
                    }
                } catch (parseErr) {
                    // Skip malformed JSON lines
                }
            }
        }

        // Add sources
        if (sources.length > 0) {
            const sourcesHtml = createSourcesHtml(sources);
            const sourcesDiv = document.createElement('div');
            sourcesDiv.innerHTML = sourcesHtml;
            assistantEl.querySelector('.message-body').appendChild(sourcesDiv);
        }
    } catch (err) {
        const typingEl = contentEl.querySelector('.typing-indicator');
        if (typingEl) typingEl.remove();
        contentEl.innerHTML = `<div class="error-message">⚠ ${err.message}</div>`;
    }

    isProcessing = false;
    sendBtn.disabled = !questionInput.value.trim();
    scrollToBottom();
}

function addMessage(role, content, isLoading = false) {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatarIcon = role === 'user'
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 8V4H8"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M6 12h12"/><rect x="2" y="8" width="20" height="12" rx="2"/></svg>';

    const bodyContent = isLoading
        ? '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>'
        : formatMarkdown(content);

    msgEl.innerHTML = `
        <div class="message-avatar">${avatarIcon}</div>
        <div class="message-body">
            <div class="message-role">${role === 'user' ? 'You' : 'Assistant'}</div>
            <div class="message-content">${bodyContent}</div>
        </div>
    `;

    messagesContainer.appendChild(msgEl);
    scrollToBottom();
    return msgEl;
}

function createSourcesHtml(sources) {
    const id = 'sources-' + Date.now();
    const cards = sources.map(s => `
        <div class="source-card">
            <div class="source-card-header">
                <span class="source-card-name">📄 ${s.source} · chunk ${s.chunk_index}</span>
                <span class="source-card-score">${(s.score * 100).toFixed(1)}% match</span>
            </div>
            <div class="source-card-text">${escapeHtml(s.text)}</div>
        </div>
    `).join('');

    return `
        <div class="sources-container">
            <button class="sources-toggle" onclick="toggleSources('${id}', this)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
                ${sources.length} source(s) used
            </button>
            <div class="sources-list" id="${id}">${cards}</div>
        </div>
    `;
}

// Global function for onclick
window.toggleSources = function (id, btn) {
    const list = document.getElementById(id);
    list.classList.toggle('visible');
    btn.classList.toggle('expanded');
};

// ── Utilities ────────────────────────────────────────────────
function formatMarkdown(text) {
    // Simple markdown formatting
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function scrollToBottom() {
    chatContainer.scrollTo({
        top: chatContainer.scrollHeight,
        behavior: 'smooth',
    });
}
