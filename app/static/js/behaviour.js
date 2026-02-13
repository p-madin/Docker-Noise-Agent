document.addEventListener("DOMContentLoaded", init);

function init() {
    // Tab Switching Initialisation
    const tabs = document.querySelectorAll('.tab');
    const liveView = document.getElementById('liveView');
    const historyView = document.getElementById('historyView');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            if (tab.dataset.tab === 'history') {
                liveView.style.display = 'none';
                historyView.classList.add('active');
                fetchHistory();
            } else {
                liveView.style.display = 'block';
                historyView.classList.remove('active');
            }
        });
    });

    // Audio Control Initialisation
    const listenBtn = document.getElementById('listenBtn');
    if (listenBtn) {
        listenBtn.addEventListener('click', toggleListening);
    }
}

const listenBtn = document.getElementById('listenBtn');
const btnText = document.getElementById('btnText');
const logContainer = document.getElementById('log');
const canvas = document.getElementById('visualizer');
const canvasCtx = canvas ? canvas.getContext('2d') : null;

const historyView = document.getElementById('historyView');
const liveView = document.getElementById('liveView');

let audioContext;
let analyser;
let microphone;
let isListening = false;
let processor;

async function fetchHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        historyView.innerHTML = '';

        if (data.length === 0) {
            historyView.innerHTML = '<div class="history-entry">No history found.</div>';
            return;
        }

        data.forEach(entry => {
            const row = document.createElement('div');
            row.className = 'history-entry';
            row.innerHTML = `
                <div>
                    <span class="label">${entry.label}</span> 
                    <span class="confidence">(Conf: ${entry.confidence.toFixed(2)})</span>
                    <div class="timestamp">${entry.timestamp}</div>
                </div>
                <div class="history-actions">
                    ${entry.has_audio ? `
                        <button class="action-btn play-btn" onclick="handleAudioClick(${entry.id}, this)" title="Play Recording">▶️</button>
                    ` : ''}
                    <button class="action-btn flag-btn ${entry.is_flagged ? 'flagged' : ''}" 
                        onclick="flagDetection(${entry.id}, this)" 
                        title="${entry.is_flagged ? 'Already Flagged' : 'Flag as Incorrect (HITL)'}">
                        ${entry.is_flagged ? '🚩' : '🏳️'}
                    </button>
                </div>
            `;
            historyView.appendChild(row);
        });
    } catch (err) {
        console.error('History fetch error:', err);
    }
}

async function handleAudioClick(id, btn) {
    // If audio is already loaded attached to this button
    if (btn.audio) {
        if (btn.audio.paused) {
            btn.audio.play();
        } else {
            btn.audio.pause();
        }
        return;
    }

    // Initial State: Load Audio
    const originalIcon = btn.innerHTML;
    // 1. Show Spinner (Spinner replaces button icon)
    btn.innerHTML = '<span class="spinner">⏳</span>';
    btn.disabled = true; // Prevent multiple clicks while loading logic creates the audio

    try {
        const response = await fetch(`/audio/${id}`);
        if (!response.ok) throw new Error('Audio not found');
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);

        // Attach audio to button to maintain state
        btn.audio = audio;

        // Setup state listeners
        audio.onplay = () => {
            btn.innerHTML = '⏸️'; // Pause button while playing
            btn.disabled = false;
        };

        audio.onpause = () => {
            btn.innerHTML = '▶️'; // Play button while paused
        };

        audio.onended = () => {
            btn.innerHTML = '▶️'; // Return to play button
            btn.audio = null;     // Reset state so next click reloads (or we could keep it to replay without fetch)
            // User requested: "once the sample is complete, the play button returns" - implies reset.
            // If we want to avoid re-fetching, we can keep btn.audio but seek to 0. 
            // But blob URLs are cheap to recreate or keep.
            // Let's explicitly clear it to match "delivered... returns", simple approach.
            URL.revokeObjectURL(url);
        };

        audio.onerror = () => {
            console.error('Audio Error');
            alert('Error playing audio');
            btn.innerHTML = originalIcon;
            btn.disabled = false;
            btn.audio = null;
        }

        // 2. Play
        await audio.play();

    } catch (err) {
        console.error('Playback error:', err);
        alert('Could not play audio: ' + err.message);
        btn.innerHTML = originalIcon;
        btn.disabled = false;
    }
}

async function flagDetection(id, btn) {
    if (btn.classList.contains('flagged')) return;

    try {
        const response = await fetch(`/flag/${id}`, { method: 'POST' });
        const data = await response.json();
        if (data.status === 'flagged') {
            btn.classList.add('flagged');
            btn.innerHTML = '🚩';
            btn.title = 'Already Flagged';
        }
    } catch (err) {
        console.error('Flagging error:', err);
    }
}

listenBtn.addEventListener('click', toggleListening);

async function toggleListening() {
    if (isListening) {
        stopListening();
    } else {
        await startListening();
    }
}

async function startListening() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        analyser = audioContext.createAnalyser();
        microphone = audioContext.createMediaStreamSource(stream);

        // For visualization
        analyser.fftSize = 256;
        microphone.connect(analyser);
        drawVisualizer();

        // For processing - capture chunks
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        microphone.connect(processor);
        processor.connect(audioContext.destination);

        let audioChunks = [];
        processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            audioChunks.push(new Float32Array(inputData));

            // Every ~0.25s per chunk. 4 chunks = 16384 samples ~ 1.024s
            if (audioChunks.length >= 4) {
                const merged = mergeBuffers(audioChunks);
                sendAudio(merged);
                audioChunks = [];
            }
        };

        isListening = true;
        listenBtn.classList.add('listening');
        btnText.textContent = 'Stop Listening';
        addLog('System', 'Microphone active. Monitoring...');

    } catch (err) {
        console.error('Error accessing microphone:', err);
        addLog('Error', 'Microphone access denied.');
    }
}

function stopListening() {
    if (processor) processor.disconnect();
    if (microphone) microphone.disconnect();
    if (audioContext) audioContext.close();

    isListening = false;
    listenBtn.classList.remove('listening');
    btnText.textContent = 'Start Listening';
    addLog('System', 'Monitoring stopped.');
}

function mergeBuffers(buffers) {
    let length = 0;
    buffers.forEach(b => length += b.length);
    let result = new Float32Array(length);
    let offset = 0;
    buffers.forEach(b => {
        result.set(b, offset);
        offset += b.length;
    });
    return result;
}

let isSending = false;

async function sendAudio(floatData) {
    if (isSending) return; // Skip if previous request is still in flight
    isSending = true;

    // Convert to 16-bit PCM WAV for simplified backend processing
    const wavBlob = createWavBlob(floatData, 16000);

    try {
        const response = await fetch('/classify_audio', {
            method: 'POST',
            body: wavBlob,
            headers: { 'X-Timestamp': new Date().toLocaleTimeString() }
        });
        const data = await response.json();
        if (data.detected !== 'Silence') {
            addLog(data.detected, data.confidence);
        }
    } catch (err) {
        console.error('Classification error:', err);
    } finally {
        isSending = false;
    }
}

function createWavBlob(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');

    // fmt sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);

    // data sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write PCM samples
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function addLog(label, value) {
    const entry = document.createElement('div');

    // Determine styling class
    let statusClass = 'Heuristic';
    if (label === 'System' || label === 'Error') {
        statusClass = label;
    } else if (!label.includes('(Heuristic)') && !label.includes('Noise') && !label.includes('Hum')) {
        statusClass = 'Detected';
    }
    entry.className = `log-entry ${statusClass}`;

    const content = document.createElement('div');
    const cleanLabel = label.replace('(Heuristic)', '').trim();

    if (typeof value === 'number') {
        content.innerHTML = `<span class="label">${cleanLabel}</span><span class="confidence">(Conf: ${value.toFixed(2)})</span>`;
    } else {
        content.innerHTML = `<span class="label">${cleanLabel}:</span> <span>${value}</span>`;
    }

    const time = document.createElement('div');
    time.className = 'timestamp';
    time.textContent = new Date().toLocaleTimeString();

    entry.appendChild(content);
    entry.appendChild(time);
    logContainer.prepend(entry);

    if (logContainer.children.length > 20) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

function drawVisualizer() {
    if (!isListening) return;
    requestAnimationFrame(drawVisualizer);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = '#0f172a';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    const barWidth = (canvas.width / bufferLength) * 2.5;
    let barHeight;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] / 2;
        canvasCtx.fillStyle = `rgb(99, 102, ${dataArray[i] + 100})`;
        canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
        x += barWidth + 1;
    }
}
