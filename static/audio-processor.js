/**
 * AudioWorkletProcessor for Real-Time PCM16 16kHz Audio Streaming.
 * Runs on a dedicated Web Audio thread for jitter-free capture.
 */
class PCM16Processor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 16000;
        // 100ms batch at 16kHz = 1600 samples
        this.batchSize = 1600;
        this.pcmBuffer = new Int16Array(this.batchSize);
        this.pcmIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }

        const channelData = input[0]; // Mono channel
        const currentSampleRate = sampleRate; // Global AudioWorklet sampleRate

        // Resample from currentSampleRate to target 16kHz
        const ratio = currentSampleRate / this.targetSampleRate;
        for (let i = 0; i < channelData.length; i += ratio) {
            const index = Math.floor(i);
            const sample = channelData[index];

            // Clamp and convert Float32 [-1.0, 1.0] to Int16 [-32768, 32767]
            const clamped = Math.max(-1.0, Math.min(1.0, sample));
            const pcm16 = clamped < 0 ? clamped * 32768 : clamped * 32767;

            this.pcmBuffer[this.pcmIndex++] = Math.round(pcm16);

            // Once 100ms batch (1600 samples = 3200 bytes) is full, send to main thread
            if (this.pcmIndex >= this.batchSize) {
                // Post transferable ArrayBuffer
                const copy = new Int16Array(this.pcmBuffer);
                this.port.postMessage(copy.buffer, [copy.buffer]);
                this.pcmIndex = 0;
            }
        }

        return true;
    }
}

registerProcessor("pcm16-processor", PCM16Processor);
