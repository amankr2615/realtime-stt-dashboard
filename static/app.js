// Hinglish STT Studio - Frontend Client Logic

let audioContext = null;
let mediaStream = null;
let micSource = null;
let workletNode = null;
let muteNode = null;
let websocket = null;
let isRecording = false;

// Telemetry state
let lastMetrics = {
    inference_ms: 0,
    rtf: 0,
    switches: 0
};

// DOM Elements
const toggleMicBtn = document.getElementById("toggleMicBtn");
const micIcon = document.getElementById("micIcon");
const micBtnText = document.getElementById("micBtnText");
const recordingIndicator = document.getElementById("recordingIndicator");
const connectionBadge = document.getElementById("connectionBadge");
const committedTextEl = document.getElementById("committedText");
const provisionalTextEl = document.getElementById("provisionalText");
const clearStreamBtn = document.getElementById("clearStreamBtn");
const saveTranscriptionBtn = document.getElementById("saveTranscriptionBtn");
const speakCurrentBtn = document.getElementById("speakCurrentBtn");
const metricLatencyEl = document.getElementById("metricLatency");
const metricRTFEl = document.getElementById("metricRTF");
const metricSwitchesEl = document.getElementById("metricSwitches");

const historyList = document.getElementById("historyList");
const historyLoading = document.getElementById("historyLoading");
const historyEmpty = document.getElementById("historyEmpty");
const searchInput = document.getElementById("searchInput");
const ttsAudioPlayer = document.getElementById("ttsAudioPlayer");
const languageModeSelect = document.getElementById("languageModeSelect");
const modelBadge = document.getElementById("modelBadge");
const historyItemsById = new Map();

// Edit Modal Elements
const editModal = document.getElementById("editModal");
const editTitleInput = document.getElementById("editTitleInput");
const editTextarea = document.getElementById("editTextarea");
const cancelEditBtn = document.getElementById("cancelEditBtn");
const saveEditBtn = document.getElementById("saveEditBtn");
let currentEditingId = null;

// Anonymous per-browser session ownership ID
function getSessionId() {
    const key = "hinglish_stt_session_id";
    let sid = localStorage.getItem(key);
    if (!sid) {
        sid = (crypto.randomUUID ? crypto.randomUUID() : `sess-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`);
        localStorage.setItem(key, sid);
    }
    return sid;
}

function apiHeaders(extra = {}) {
    return { "X-Session-ID": getSessionId(), ...extra };
}

// -------------------------------------------------------------
// WebSocket & Audio Capture
// -------------------------------------------------------------

function getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/transcribe`;
}

function updateConnectionStatus(status) {
    if (status === "connected") {
        connectionBadge.className = "inline-flex items-center px-2.5 py-1.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200";
        connectionBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span>Live Connected`;
    } else if (status === "connecting") {
        connectionBadge.className = "inline-flex items-center px-2.5 py-1.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200";
        connectionBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5"></span>Connecting...`;
    } else {
        connectionBadge.className = "inline-flex items-center px-2.5 py-1.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600 border border-gray-200";
        connectionBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-gray-400 mr-1.5"></span>Disconnected`;
    }
}

async function startAudioCapture() {
    try {
        updateConnectionStatus("connecting");
        
        // 1. Establish WebSocket connection
        websocket = new WebSocket(getWebSocketUrl());
        websocket.binaryType = "arraybuffer";

        websocket.onopen = () => {
            updateConnectionStatus("connected");
            // Sync selected language mode with backend
            if (languageModeSelect) {
                websocket.send(JSON.stringify({
                    action: "set_language",
                    language: languageModeSelect.value
                }));
            }
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "partial") {
                committedTextEl.textContent = data.committed ? data.committed + " " : "";
                provisionalTextEl.textContent = data.provisional || "";
                
                // Update metrics HUD
                if (data.metrics) {
                    lastMetrics = data.metrics;
                    metricLatencyEl.textContent = `${data.metrics.inference_ms} ms`;
                    metricRTFEl.textContent = data.metrics.rtf;
                    metricSwitchesEl.textContent = data.metrics.switches;
                }
            } else if (data.type === "final") {
                committedTextEl.textContent = data.text ? data.text + " " : "";
                provisionalTextEl.textContent = "";
                if (data.metrics) {
                    if (data.metrics.p50_ms) metricLatencyEl.textContent = `${data.metrics.p50_ms} ms`;
                    if (data.metrics.switches !== undefined) metricSwitchesEl.textContent = data.metrics.switches;
                }
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.close();
                }
            }
        };

        websocket.onerror = (err) => {
            console.error("WebSocket error:", err);
            updateConnectionStatus("disconnected");
        };

        websocket.onclose = () => {
            updateConnectionStatus("disconnected");
        };

        // 2. Setup AudioContext at 16kHz — the browser's built-in sinc resampler
        //    converts device rate (usually 44.1kHz or 48kHz) to 16kHz internally.
        //    This is MUCH higher quality than manual resampling in the worklet.
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        await audioContext.resume();

        // Request mono mic with browser-level noise suppression and echo cancellation.
        // Do NOT force sampleRate in getUserMedia constraints — let the browser pick
        // the best native device rate and resample via AudioContext.sampleRate.
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                // Do NOT set sampleRate here — forces device into a mode it may not support
            }
        });

        await audioContext.audioWorklet.addModule("/static/audio-processor.js");
        micSource = audioContext.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(audioContext, "pcm16-processor");
        muteNode = audioContext.createGain();
        muteNode.gain.value = 0;

        workletNode.port.onmessage = (event) => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(event.data);
            }
        };

        micSource.connect(workletNode);
        workletNode.connect(muteNode);
        muteNode.connect(audioContext.destination);

        // Update UI
        isRecording = true;
        toggleMicBtn.className = "min-h-[44px] flex items-center justify-center space-x-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 bg-rose-600 hover:bg-rose-700 text-white shadow-sm recording-pulse";
        micIcon.textContent = "⏹️";
        micBtnText.textContent = "Stop Recording";
        recordingIndicator.classList.remove("hidden");

        if (provisionalTextEl.textContent.includes("Click \"Start Recording\"")) {
            provisionalTextEl.textContent = "Listening...";
        }

    } catch (err) {
        console.error("Failed to start audio stream:", err);
        alert(`Microphone access error: ${err.message}`);
        stopAudioCapture();
    }
}

function stopAudioCapture() {
    isRecording = false;

    // Send finalize action to WebSocket and wait for server's "final" response (up to 5s fallback)
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ action: "finalize" }));
        const wsRef = websocket;
        setTimeout(() => {
            if (wsRef && wsRef.readyState === WebSocket.OPEN) wsRef.close();
        }, 5000);
    }

    if (workletNode) {
        workletNode.port.onmessage = null;
        workletNode.disconnect();
        workletNode = null;
    }

    if (muteNode) {
        muteNode.disconnect();
        muteNode = null;
    }

    if (micSource) {
        micSource.disconnect();
        micSource = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    if (audioContext && audioContext.state !== "closed") {
        audioContext.close();
        audioContext = null;
    }

    // Reset UI
    toggleMicBtn.className = "min-h-[44px] flex items-center justify-center space-x-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm active:scale-[0.98]";
    micIcon.textContent = "🎙️";
    micBtnText.textContent = "Start Recording";
    recordingIndicator.classList.add("hidden");
    updateConnectionStatus("disconnected");
}

toggleMicBtn.addEventListener("click", () => {
    if (!isRecording) {
        startAudioCapture();
    } else {
        stopAudioCapture();
    }
});

clearStreamBtn.addEventListener("click", () => {
    committedTextEl.textContent = "";
    provisionalTextEl.textContent = "Click \"Start Recording\" and speak in Hindi, English, or Hinglish...";
    metricLatencyEl.textContent = "-- ms";
    metricRTFEl.textContent = "--";
    metricSwitchesEl.textContent = "0";
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ action: "reset" }));
    }
});

// -------------------------------------------------------------
// CRUD Operations (Database History)
// -------------------------------------------------------------

async function fetchHistory(query = "") {
    try {
        historyLoading.classList.remove("hidden");
        historyEmpty.classList.add("hidden");
        historyList.classList.add("hidden");

        const url = query ? `/api/transcriptions?search=${encodeURIComponent(query)}` : "/api/transcriptions";
        const res = await fetch(url, { headers: apiHeaders() });
        if (!res.ok) throw new Error("Failed to load records");
        const items = await res.json();

        historyLoading.classList.add("hidden");

        if (items.length === 0) {
            historyEmpty.classList.remove("hidden");
            historyList.innerHTML = "";
            return;
        }

        historyEmpty.classList.add("hidden");
        historyList.classList.remove("hidden");
        renderHistoryItems(items);
    } catch (err) {
        console.error("Error fetching history:", err);
        historyLoading.textContent = "Error loading history from database.";
    }
}

function renderHistoryItems(items) {
    historyItemsById.clear();
    items.forEach(item => historyItemsById.set(Number(item.id), item));

    historyList.innerHTML = items.map(item => {
        const safeId = Number(item.id);
        const date = escapeHtml(new Date(item.created_at).toLocaleString());
        const metrics = item.metrics || {};
        const latencyBadge = metrics.p50_ms ? `${Number(metrics.p50_ms)}ms` : (metrics.inference_ms ? `${Number(metrics.inference_ms)}ms` : "");
        const switchesBadge = metrics.switches !== undefined ? `${Number(metrics.switches)} switches` : "";

        return `
            <div class="p-4 sm:p-5 hover:bg-gray-50 transition-colors flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4" id="item-${safeId}">
                <div class="space-y-1.5 flex-1 min-w-0">
                    <div class="flex flex-wrap items-center gap-2">
                        <h4 class="text-sm font-semibold text-gray-900">${escapeHtml(item.title)}</h4>
                        <span class="px-2 py-0.5 rounded text-[11px] font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">${escapeHtml(item.language)}</span>
                        ${latencyBadge ? `<span class="px-1.5 py-0.5 rounded text-[11px] font-mono bg-gray-100 text-gray-700 border border-gray-200">${escapeHtml(latencyBadge)}</span>` : ""}
                        ${switchesBadge ? `<span class="px-1.5 py-0.5 rounded text-[11px] font-mono bg-amber-50 text-amber-700 border border-amber-200">${escapeHtml(switchesBadge)}</span>` : ""}
                    </div>
                    <p class="text-sm text-gray-700 leading-relaxed break-words">${escapeHtml(item.text)}</p>
                    <div class="text-[11px] text-gray-400 font-mono">${date}</div>
                </div>

                <div class="flex flex-wrap items-center gap-2 shrink-0 w-full sm:w-auto">
                    <button type="button" data-action="tts" data-id="${safeId}"
                        class="min-h-[36px] px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-medium transition-colors" title="Listen (TTS)">
                        🔊 Listen
                    </button>
                    <button type="button" data-action="edit" data-id="${safeId}"
                        class="min-h-[36px] px-3 py-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-medium transition-colors" title="Edit">
                        ✏️ Edit
                    </button>
                    <button type="button" data-action="delete" data-id="${safeId}"
                        class="min-h-[36px] px-3 py-1.5 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-medium border border-rose-200 transition-colors" title="Delete">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
    }).join("");
}

historyList.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const id = Number(btn.dataset.id);
    const item = historyItemsById.get(id);
    if (!item) return;

    const action = btn.dataset.action;
    if (action === "tts") {
        playTTS(id, item.text);
    } else if (action === "edit") {
        openEditModal(id, item.title, item.text);
    } else if (action === "delete") {
        deleteItem(id);
    }
});

// Save Current Canvas to Database
saveTranscriptionBtn.addEventListener("click", async () => {
    const fullText = (committedTextEl.textContent + " " + provisionalTextEl.textContent).trim();
    if (!fullText || fullText.includes("Click \"Start Recording\"")) {
        alert("Please record or speak some text first before saving.");
        return;
    }

    try {
        saveTranscriptionBtn.disabled = true;
        saveTranscriptionBtn.textContent = "Saving...";

        const payload = {
            title: `Hinglish Note - ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`,
            text: fullText,
            language: "Hinglish",
            duration_seconds: 0.0,
            metrics: lastMetrics
        };

        const res = await fetch("/api/transcriptions", {
            method: "POST",
            headers: apiHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Failed to save record");
        
        saveTranscriptionBtn.textContent = "Saved ✓";
        setTimeout(() => {
            saveTranscriptionBtn.disabled = false;
            saveTranscriptionBtn.innerHTML = `<span>💾</span><span>Save to History</span>`;
        }, 1200);

        // Refresh History
        fetchHistory();
    } catch (err) {
        console.error("Save error:", err);
        alert(`Failed to save: ${err.message}`);
        saveTranscriptionBtn.disabled = false;
        saveTranscriptionBtn.innerHTML = `<span>💾</span><span>Save to History</span>`;
    }
});

// Edit Modal Handling
window.openEditModal = function(id, title, text) {
    currentEditingId = id;
    editTitleInput.value = title;
    editTextarea.value = text;
    editModal.classList.remove("hidden");
};

cancelEditBtn.addEventListener("click", () => {
    editModal.classList.add("hidden");
    currentEditingId = null;
});

saveEditBtn.addEventListener("click", async () => {
    if (!currentEditingId) return;

    try {
        const title = editTitleInput.value.trim();
        const text = editTextarea.value.trim();
        if (!text) {
            alert("Text cannot be empty.");
            return;
        }

        const res = await fetch(`/api/transcriptions/${currentEditingId}`, {
            method: "PUT",
            headers: apiHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ title, text })
        });

        if (!res.ok) throw new Error("Update failed");

        editModal.classList.add("hidden");
        currentEditingId = null;
        fetchHistory(searchInput.value.trim());
    } catch (err) {
        console.error("Update error:", err);
        alert(`Failed to update: ${err.message}`);
    }
});

// Delete Record
window.deleteItem = async function(id) {
    if (!confirm("Are you sure you want to delete this transcription record?")) return;

    try {
        const res = await fetch(`/api/transcriptions/${id}`, {
            method: "DELETE",
            headers: apiHeaders()
        });
        if (!res.ok) throw new Error("Failed to delete record");
        
        // Optimistic DOM removal
        const row = document.getElementById(`item-${id}`);
        if (row) row.remove();
        
        if (historyList.children.length === 0) {
            historyEmpty.classList.remove("hidden");
            historyList.classList.add("hidden");
        }
    } catch (err) {
        console.error("Delete error:", err);
        alert(`Delete failed: ${err.message}`);
    }
};

// Search handling with debounce
let searchTimer = null;
searchInput.addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
        fetchHistory(e.target.value.trim());
    }, 300);
});

// -------------------------------------------------------------
// Text-to-Speech (TTS)
// Primary: Browser Web Speech API (zero latency, works offline, supports hi-IN)
// Fallback: Backend edge-tts endpoint (requires network)
// -------------------------------------------------------------

function speakWithBrowser(text) {
    if (!('speechSynthesis' in window)) return false;
    window.speechSynthesis.cancel(); // stop any ongoing speech

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "hi-IN";
    utterance.rate = 0.9;
    utterance.pitch = 1.0;

    // Prefer a Hindi voice if available
    const voices = window.speechSynthesis.getVoices();
    const hindiVoice = voices.find(v => v.lang === "hi-IN" || v.lang.startsWith("hi"));
    if (hindiVoice) utterance.voice = hindiVoice;

    window.speechSynthesis.speak(utterance);
    return true;
}

window.playTTS = async function(id, text) {
    if (!text.trim()) return;
    // Always try browser-native first — it's instant and works offline
    if (speakWithBrowser(text)) return;

    // Fallback: backend edge-tts (requires internet)
    try {
        const res = await fetch("/api/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        if (!res.ok) throw new Error("Backend TTS failed");
        const blob = await res.blob();
        const audioUrl = URL.createObjectURL(blob);
        ttsAudioPlayer.src = audioUrl;
        ttsAudioPlayer.play();
    } catch (err) {
        console.warn("TTS unavailable:", err);
    }
};

speakCurrentBtn.addEventListener("click", () => {
    const fullText = (committedTextEl.textContent + " " + provisionalTextEl.textContent).trim();
    if (!fullText || fullText.includes("Click \"Start Recording\"")) {
        alert("Nothing to speak yet.");
        return;
    }
    playTTS(0, fullText);
});

// Utility
function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

async function loadModelBadge() {
    if (!modelBadge) return;
    try {
        const res = await fetch("/api/health");
        if (!res.ok) return;
        const data = await res.json();
        if (data.model) {
            modelBadge.textContent = `faster-whisper (${data.model})`;
        }
    } catch (_) {
        // Keep default badge text if health check fails
    }
}

if (languageModeSelect) {
    languageModeSelect.addEventListener("change", (e) => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                action: "set_language",
                language: e.target.value
            }));
        }
    });
}

// Initial Load
loadModelBadge();
fetchHistory();
