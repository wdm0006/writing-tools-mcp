from server.server import reading_time
import textstat as ts

def test_reading_time_is_unusually_low_for_long_text():
    phrase = "Lorem ipsum dolor sit amet "
    # Approximately 1735 words
    text_long = phrase * 347
    num_words = len(text_long.split())

    # Typical WPM reading speeds
    typical_wpm = 250

    expected_time_lower_typical_minutes = num_words / typical_wpm

    result = reading_time(text_long, level="full")
    estimated_tool_minutes = result.get("full_text")

    assert estimated_tool_minutes is not None, "Reading time should be calculated."
    
    assert estimated_tool_minutes > (expected_time_lower_typical_minutes / 2)
    assert estimated_tool_minutes < (expected_time_lower_typical_minutes * 2)