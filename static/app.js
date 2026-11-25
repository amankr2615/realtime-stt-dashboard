// Hinglish STT Studio - Frontend Client Logic

let audioContext = null;
let mediaStream = null;
let workletNode = null;
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

// Edit Modal Elements
const editModal = document.getElementById("editModal");
const editTitleInput = document.getElementById("editTitleInput");
const editTextarea = document.getElementById("editTextarea");
const cancelEditBtn = document.getElementById("cancelEditBtn");
const saveEditBtn = document.getElementById("saveEditBtn");
let currentEditingId = null;

// -------------------------------------------------------------
// WebSocket & Audio Capture
// -------------------------------------------------------------

function getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/transcribe`;
}

function updateConnectionStatus(status) {
    if (status === "connected") {
        connectionBadge.className = "inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-950/80 text-emerald-400 border border-emerald-800/60";
        connectionBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1.5 animate-pulse"></span>Live Connected`;
    } else if (status === "connecting") {
        connectionBadge.className = "inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-amber-950/80 text-amber-400 border border-amber-800/60";
        connectionBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5"></span>Connecting...`;
    } else {
        connectionBadge.className = "inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-400 border border-slate-700";
        connectionBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-slate-500 mr-1.5"></span>Disconnected`;
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
        const micSource = audioContext.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(audioContext, "pcm16-processor");

        workletNode.port.onmessage = (event) => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(event.data);
            }
        };

        micSource.connect(workletNode);
        // Do NOT connect workletNode to speakers — that would create a feedback loop

        // Update UI
        isRecording = true;
        toggleMicBtn.className = "flex items-center space-x-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 recording-pulse";
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

    // Send finalize action to WebSocket
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ action: "finalize" }));
        setTimeout(() => {
            if (websocket) websocket.close();
        }, 300);
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
    toggleMicBtn.className = "flex items-center space-x-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/30 active:scale-95";
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
        const res = await fetch(url);
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
    historyList.innerHTML = items.map(item => {
        const date = new Date(item.created_at).toLocaleString();
        const metrics = item.metrics || {};
        const latencyBadge = metrics.p50_ms ? `${metrics.p50_ms}ms` : (metrics.inference_ms ? `${metrics.inference_ms}ms` : "");
        const switchesBadge = metrics.switches !== undefined ? `${metrics.switches} switches` : "";

        return `
            <div class="p-5 hover:bg-slate-800/40 transition-colors flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4" id="item-${item.id}">
                <div class="space-y-1.5 flex-1 pr-4">
                    <div class="flex items-center space-x-2">
                        <h4 class="text-sm font-semibold text-white">${escapeHtml(item.title)}</h4>
                        <span class="px-2 py-0.5 rounded text-[10px] font-medium bg-indigo-950 text-indigo-300 border border-indigo-800/60">${item.language}</span>
                        ${latencyBadge ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300">${latencyBadge}</span>` : ""}
                        ${switchesBadge ? `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-950/80 text-amber-300 border border-amber-800/50">${switchesBadge}</span>` : ""}
                    </div>
                    <p class="text-sm text-slate-300 leading-relaxed">${escapeHtml(item.text)}</p>
                    <div class="text-[11px] text-slate-500 font-mono">${date}</div>
                </div>

                <div class="flex items-center space-x-2 shrink-0">
                    <button onclick="playTTS(${item.id}, ${JSON.stringify(item.text).replace(/"/g, '&quot;')})" 
                        class="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition-colors" title="Listen (TTS)">
                        🔊 Listen
                    </button>
                    <button onclick="openEditModal(${item.id}, ${JSON.stringify(item.title).replace(/"/g, '&quot;')}, ${JSON.stringify(item.text).replace(/"/g, '&quot;')})" 
                        class="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition-colors" title="Edit">
                        ✏️ Edit
                    </button>
                    <button onclick="deleteItem(${item.id})" 
                        class="p-2 rounded-lg bg-rose-950/60 hover:bg-rose-900/80 text-rose-300 text-xs border border-rose-800/40 transition-colors" title="Delete">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
    }).join("");
}

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
            headers: { "Content-Type": "application/json" },
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
            headers: { "Content-Type": "application/json" },
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
        const res = await fetch(`/api/transcriptions/${id}`, { method: "DELETE" });
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
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
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
fetchHistory();
