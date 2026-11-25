import pytest
from backend.transcriber import count_language_switches, StreamingTranscriber

def test_count_language_switches():
    # Pure English -> 0 switches
    assert count_language_switches("hello world how are you") == 0
    
    # Pure Hindi Devanagari -> 0 switches
    assert count_language_switches("नमस्ते आप कैसे हैं") == 0
    
    # Code-switched Hinglish: "आज" (Dev) -> "meeting" (Lat) -> "में" (Dev) -> "discuss" (Lat) -> "करेंगे" (Dev)
    # Switches: 4 transitions
    text = "आज meeting में discuss करेंगे"
    assert count_language_switches(text) == 4

def test_streaming_transcriber_agreement():
    transcriber = StreamingTranscriber()
    
    # Pass 1: "main kal"
    res1 = transcriber.process_hypothesis("main kal", audio_duration_sec=1.0, inference_time_sec=0.1)
    # No prior hypothesis to agree with, so committed is empty, provisional is "main kal"
    assert res1["committed_text"] == ""
    assert res1["provisional_text"] == "main kal"
    
    # Pass 2: "main kal office jaunga"
    res2 = transcriber.process_hypothesis("main kal office jaunga", audio_duration_sec=2.0, inference_time_sec=0.12)
    # "main kal" matches both passes -> committed!
    assert "main kal" in res2["committed_text"]
    assert "office jaunga" in res2["provisional_text"]
    
    # Pass 3: finalize
    res_final = transcriber.finalize()
    assert res_final["final_text"] == "main kal office jaunga"


def test_rtf_distinguishes_window_from_stream():
    transcriber = StreamingTranscriber()
    # Rolling window is 4.0s, but only 1.0s of new audio arrived since last pass, and inference took 0.5s
    res = transcriber.process_hypothesis(
        "hello world",
        audio_duration_sec=4.0,
        inference_time_sec=0.5,
        new_audio_sec=1.0,
    )
    assert res["metrics"]["window_rtf"] == pytest.approx(0.125, abs=0.001)  # 0.5 / 4.0
    assert res["metrics"]["rtf"] == pytest.approx(0.5, abs=0.001)          # 0.5 / 1.0


def test_reset_clears_transcription_and_metrics():
    transcriber = StreamingTranscriber()
    transcriber.process_hypothesis("aaj meeting hai", audio_duration_sec=1.0, inference_time_sec=0.2)
    transcriber.process_hypothesis("aaj meeting hai", audio_duration_sec=1.0, inference_time_sec=0.2)
    assert len(transcriber.committed_words) > 0

    transcriber.reset()
    assert transcriber.committed_words == []
    assert transcriber.last_hypothesis_words == []
    assert len(transcriber.inference_latencies) == 0
    assert transcriber.total_received_audio_sec == 0.0
    assert transcriber.total_inference_sec == 0.0


def test_latency_history_bounded():
    transcriber = StreamingTranscriber(max_latency_samples=5)
    for i in range(12):
        transcriber.process_hypothesis(f"word{i}", audio_duration_sec=1.0, inference_time_sec=0.1)
    assert len(transcriber.inference_latencies) == 5


def test_empty_inference_counted_in_rtf():
    transcriber = StreamingTranscriber()
    # Pass 1: empty hypothesis still consumes 0.5s inference for 1.0s new audio
    transcriber.record_empty_inference(inference_time_sec=0.5, new_audio_sec=1.0)
    # Pass 2: 0.5s inference for 1.0s new audio -> total 1.0s infer / 2.0s audio = 0.5 RTF
    res = transcriber.process_hypothesis("hello", audio_duration_sec=2.0, inference_time_sec=0.5, new_audio_sec=1.0)
    assert res["metrics"]["rtf"] == pytest.approx(0.5, abs=0.001)
    assert len(transcriber.inference_latencies) == 2
