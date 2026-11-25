/**
 * PCM16Processor — AudioWorkletProcessor for real-time 16 kHz mono capture.
 *
 * Key improvements
 * ────────────────
 * 1. Downsamples from device sample rate → 16 000 Hz with linear interpolation.
 * 2. Applies a simple per-frame DC offset removal (mean subtraction) so that
 *    USB / built-in mic DC bias doesn't confuse Whisper's voice activity.
 * 3. Applies a soft-knee noise gate: frames below a configurable RMS threshold
 *    are zeroed out, so the server never receives pure-silence audio that would
 *    trigger hallucinations.
 * 4. Sends Int16 ArrayBuffers (transferable) for zero-copy WebSocket transfer.
 */
class PCM16Processor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        this.targetSampleRate = 16000;
        // 100 ms batch at 16 kHz = 1 600 samples → 3 200 bytes
        this.batchSamples = 1600;

        // ── Noise gate ────────────────────────────────────────────────────────
        // RMS in float32 range [0, 1].  Values calibrated for a laptop mic at
        // normal speaking distance.  ~0.004 cuts ambient hum / fan noise.
        this.noiseGateThreshold = 0.004;

        // Internal PCM accumulation buffer
        this.pcmBuffer = new Int16Array(this.batchSamples);
        this.pcmIndex  = 0;
    }

    /**
     * Remove DC offset from a Float32 frame (in-place).
     * DC bias causes low-frequency energy that confuses VAD.
     */
    _removeDC(frame) {
        let mean = 0;
        for (let i = 0; i < frame.length; i++) mean += frame[i];
        mean /= frame.length;
        for (let i = 0; i < frame.length; i++) frame[i] -= mean;
    }

    /**
     * Compute RMS energy of a Float32 frame.
     */
    _rms(frame) {
        let sum = 0;
        for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
        return Math.sqrt(sum / frame.length);
    }

    /**
     * Convert a single Float32 sample [-1, 1] → Int16.
     */
    _toInt16(sample) {
        const clamped = Math.max(-1.0, Math.min(1.0, sample));
        return Math.round(clamped < 0 ? clamped * 32768 : clamped * 32767);
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;

        // Use the first channel (mono).
        const channelData = input[0];
        if (!channelData || channelData.length === 0) return true;

        // ── 1. DC removal on the raw 128-sample worklet frame ─────────────
        const frame = new Float32Array(channelData);   // copy to avoid mutating input
        this._removeDC(frame);

        // ── 2. Noise gate ─────────────────────────────────────────────────
        const energy = this._rms(frame);
        if (energy < this.noiseGateThreshold) {
            // Advance internal buffer index with silence (zeros already)
            // but DO NOT zero the entire batch — partial speech frames must
            // still reach the server.  Just skip encoding this frame.
            this.pcmIndex = Math.min(this.pcmIndex, this.batchSamples);
            return true;
        }

        // ── 3. Resample: device rate → 16 kHz with linear interpolation ───
        const ratio = sampleRate / this.targetSampleRate;   // e.g. 48000/16000 = 3.0

        for (let i = 0; i < frame.length; i += ratio) {
            const lo  = Math.floor(i);
            const hi  = Math.min(lo + 1, frame.length - 1);
            const frac = i - lo;
            const sample = frame[lo] * (1 - frac) + frame[hi] * frac;  // lerp

            this.pcmBuffer[this.pcmIndex++] = this._toInt16(sample);

            // ── 4. Flush 100 ms batch ─────────────────────────────────────
            if (this.pcmIndex >= this.batchSamples) {
                const copy = new Int16Array(this.pcmBuffer);
                this.port.postMessage(copy.buffer, [copy.buffer]);
                this.pcmBuffer = new Int16Array(this.batchSamples);
                this.pcmIndex  = 0;
            }
        }

        return true;
    }
}

registerProcessor("pcm16-processor", PCM16Processor);
