// ── State ──────────────────────────────────────────────────────────────────────
let pendingFiles = [];
let isStreaming  = false;

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadStatus();
  setupDropZone();
  autoResizeTextarea();
});

// ── Status ─────────────────────────────────────────────────────────────────────
async function loadStatus() {
  try {
    const res  = await fetch("/status");
    const data = await res.json();
    updateStatus(data);
  } catch (e) {
    console.error("Status fetch failed:", e);
  }
}

function updateStatus(data) {
  const badge    = document.getElementById("dbStatus");
  const clearBtn = document.getElementById("clearBtn");
  const indexed  = document.getElementById("indexedSection");
  const fileDiv  = document.getElementById("indexedFiles");
  const empty    = document.getElementById("emptyState");

  if (data.chunk_count > 0) {
    badge.className  = "status-badge success";
    badge.textContent = `💾 Pinecone: ${data.chunk_count} vectors stored`;
    clearBtn.style.display = "block";
  } else {
    badge.className  = "status-badge empty";
    badge.textContent = "💾 Pinecone: empty";
    clearBtn.style.display = "none";
  }

  if (data.indexed_files && data.indexed_files.length > 0) {
    indexed.style.display = "block";
    fileDiv.innerHTML = data.indexed_files
      .map(f => `<div class="indexed-file">• ${f}</div>`)
      .join("");
  } else {
    indexed.style.display = "none";
  }

  if (data.ready) {
    if (empty) empty.style.display = "none";
  }
}

// ── File handling ───────────────────────────────────────────────────────────────
function setupDropZone() {
  const zone  = document.getElementById("dropZone");
  const input = document.getElementById("fileInput");

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    addFiles([...e.dataTransfer.files]);
  });

  input.addEventListener("change", () => addFiles([...input.files]));
}

function addFiles(files) {
  const allowed = ["pdf","txt","md","docx","csv","xlsx","json","pptx"];
  for (const f of files) {
    const ext = f.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
      showToast(`${f.name}: unsupported type`, "error");
      continue;
    }
    if (f.size > 20 * 1024 * 1024) {
      showToast(`${f.name}: exceeds 20MB`, "error");
      continue;
    }
    if (!pendingFiles.find(p => p.name === f.name)) {
      pendingFiles.push(f);
    }
  }
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById("fileList");
  list.innerHTML = pendingFiles.map((f, i) => `
    <div class="file-item">
      <span>📄 ${f.name}</span>
      <span class="remove" onclick="removeFile(${i})">✕</span>
    </div>
  `).join("");
}

function removeFile(i) {
  pendingFiles.splice(i, 1);
  renderFileList();
}

// ── Build index ─────────────────────────────────────────────────────────────────
async function buildIndex() {
  if (pendingFiles.length === 0) {
    showToast("Please upload at least one document.", "error");
    return;
  }

  const btn = document.getElementById("buildBtn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Uploading...';

  // Upload files
  const formData = new FormData();
  for (const f of pendingFiles) formData.append("files", f);

  try {
    const upRes  = await fetch("/upload", { method: "POST", body: formData });
    const upData = await upRes.json();

    if (upData.errors && upData.errors.length > 0) {
      showToast(upData.errors.join("\n"), "error");
    }

    if (!upData.saved || upData.saved.length === 0) {
      showToast("No files were uploaded.", "error");
      btn.disabled = false;
      btn.textContent = "🔨 Build Index";
      return;
    }

    btn.innerHTML = '<span class="spinner"></span> Building index...';

    const buildRes  = await fetch("/build", { method: "POST" });
    const buildData = await buildRes.json();

    if (!buildRes.ok) {
      showToast(buildData.detail || "Build failed.", "error");
      btn.disabled = false;
      btn.textContent = "🔨 Build Index";
      return;
    }

    pendingFiles = [];
    renderFileList();
    updateStatus({ ...buildData, ready: true });
    showToast(`✅ Index built — ${buildData.chunk_count} vectors stored`, "success");

    const empty = document.getElementById("emptyState");
    if (empty) empty.style.display = "none";

  } catch (e) {
    showToast("Build failed: " + e.message, "error");
  }

  btn.disabled = false;
  btn.textContent = "🔨 Build Index";
}

// ── Clear database ──────────────────────────────────────────────────────────────
async function clearDatabase() {
  if (!confirm("Clear all indexed documents? This cannot be undone.")) return;
  try {
    await fetch("/clear", { method: "POST" });
    updateStatus({ chunk_count: 0, indexed_files: [], ready: false });
    document.getElementById("messages").innerHTML = `
      <div class="empty-state" id="emptyState">
        <p>👈 Upload documents and click <strong>Build Index</strong> to get started.</p>
      </div>`;
    showToast("Database cleared.", "success");
  } catch (e) {
    showToast("Clear failed: " + e.message, "error");
  }
}

// ── Chat ────────────────────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function sendMessage() {
  const input = document.getElementById("userInput");
  const msg   = input.value.trim();
  if (!msg || isStreaming) return;

  input.value = "";
  input.style.height = "auto";
  isStreaming = true;

  const sendBtn = document.getElementById("sendBtn");
  sendBtn.disabled = true;

  // Remove empty state
  const empty = document.getElementById("emptyState");
  if (empty) empty.remove();

  // Add user message
  appendMessage("user", msg);

  // Add assistant bubble with typing indicator
  const assistantId = "msg-" + Date.now();
  appendAssistantBubble(assistantId);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });

    if (!res.ok) {
      const err = await res.json();
      updateBubble(assistantId, "⚠️ " + (err.detail || "Error occurred."), []);
      isStreaming = false;
      sendBtn.disabled = false;
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";
    let   fullText = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === "text") {
            fullText += event.content;
            updateBubbleText(assistantId, fullText);
          } else if (event.type === "done") {
            updateBubble(assistantId, fullText, event.sources || []);
          } else if (event.type === "error") {
            updateBubble(assistantId, "⚠️ " + event.content, []);
          }
        } catch {}
      }
    }
  } catch (e) {
    updateBubble(assistantId, "⚠️ Connection error: " + e.message, []);
  }

  isStreaming = false;
  sendBtn.disabled = false;
  scrollToBottom();
}

// ── DOM helpers ─────────────────────────────────────────────────────────────────
function appendMessage(role, content) {
  const messages = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="avatar">${role === "user" ? "👤" : "🤖"}</div>
    <div class="bubble">${escapeHtml(content)}</div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

function appendAssistantBubble(id) {
  const messages = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = "message assistant";
  div.id = id;
  div.innerHTML = `
    <div class="avatar">🤖</div>
    <div class="bubble">
      <div class="typing"><span></span><span></span><span></span></div>
    </div>
  `;
  messages.appendChild(div);
  scrollToBottom();
}

function updateBubbleText(id, text) {
  const el = document.getElementById(id);
  if (!el) return;
  const bubble = el.querySelector(".bubble");
  bubble.innerHTML = formatText(text);
  scrollToBottom();
}

function updateBubble(id, text, sources) {
  const el = document.getElementById(id);
  if (!el) return;
  const bubble = el.querySelector(".bubble");

  let html = formatText(text);

  if (sources && sources.length > 0) {
    html += `<div class="sources">
      <div class="sources-title">📎 Sources</div>
      ${sources.map(s => `
        <div class="source-item">
          📄 <strong>${escapeHtml(s.file)}</strong>
          ${s.page ? `— p. ${s.page}` : ""}
          <span class="score">(relevance: ${s.score})</span>
        </div>
      `).join("")}
    </div>`;
  }

  bubble.innerHTML = html;
  scrollToBottom();
}

function formatText(text) {
  return escapeHtml(text)
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^/, "<p>")
    .replace(/$/, "</p>");
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scrollToBottom() {
  const messages = document.getElementById("messages");
  messages.scrollTop = messages.scrollHeight;
}

function autoResizeTextarea() {
  const ta = document.getElementById("userInput");
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  });
}

// ── Toast ───────────────────────────────────────────────────────────────────────
function showToast(msg, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}