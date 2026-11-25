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
