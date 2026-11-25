/**
 * PCM16Processor — High-quality 16 kHz PCM16 AudioWorkletProcessor
 *
 * Strategy: Let the browser's AudioContext handle resampling natively.
 * We create the AudioContext with sampleRate:16000 directly — on Chrome/Safari
 * on macOS the browser resamples from the device rate (48kHz) using its own
 * high-quality sinc filter BEFORE the samples reach this worklet.
 * This eliminates all manual resampling artifacts.
 *
 * What this processor does:
 * 1. Receives already-16kHz Float32 frames from the browser resampler
 * 2. Applies DC offset removal per 128-sample frame
 * 3. Applies a soft noise gate (zeros near-silence frames)
 * 4. Packs samples into Int16 and sends 100ms batches over the port
 */
class PCM16Processor extends AudioWorkletProcessor {
    constructor() {
        super();
        // At 16kHz, 100ms = 1600 samples
        this.BATCH_SAMPLES = 1600;
        this.pcmBuffer     = new Int16Array(this.BATCH_SAMPLES);
        this.writeIndex    = 0;

        // Noise gate: silence threshold in float32 RMS (0 to 1)
        // 0.0008 cuts dead digital silence while allowing quiet smartphone/laptop voices
        this.NOISE_GATE_RMS = 0.0008;
    }

    _rms(samples) {
        let sum = 0;
        for (let i = 0; i < samples.length; i++) {
            sum += samples[i] * samples[i];
        }
        return Math.sqrt(sum / samples.length);
    }

    _removeDC(samples) {
        // Per-frame mean subtraction removes low-frequency DC bias from USB/built-in mics
        let mean = 0;
        for (let i = 0; i < samples.length; i++) mean += samples[i];
        mean /= samples.length;
        for (let i = 0; i < samples.length; i++) samples[i] -= mean;
    }

    process(inputs) {
        const channel = inputs[0]?.[0];
        if (!channel || channel.length === 0) return true;

        // Work on a mutable copy
        const frame = new Float32Array(channel);

        // 1. Remove DC offset
        this._removeDC(frame);

        // 2. Noise gate — if this 128-sample frame is too quiet, skip it
        //    This prevents sending silence to the server which triggers hallucinations
        if (this._rms(frame) < this.NOISE_GATE_RMS) {
            return true;
        }

        // 3. Convert each sample Float32 → Int16 and accumulate
        for (let i = 0; i < frame.length; i++) {
            const s = Math.max(-1.0, Math.min(1.0, frame[i]));
            this.pcmBuffer[this.writeIndex++] = Math.round(s < 0 ? s * 32768 : s * 32767);

            // 4. Flush a complete 100ms batch
            if (this.writeIndex >= this.BATCH_SAMPLES) {
                const copy = new Int16Array(this.pcmBuffer);
                this.port.postMessage(copy.buffer, [copy.buffer]);
                this.pcmBuffer  = new Int16Array(this.BATCH_SAMPLES);
                this.writeIndex = 0;
            }
        }

        return true;
    }
}

registerProcessor("pcm16-processor", PCM16Processor);
