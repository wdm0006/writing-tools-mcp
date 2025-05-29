"""Readability analysis functionality."""

import textstat

from server.text_processing import parse_markdown_sections, split_paragraphs, strip_markdown_markup


class ReadabilityAnalyzer:
    """Handles readability scoring and reading time estimation."""

    def __init__(self):
        # Default reading speed configuration
        self.ms_per_character = 28

    def readability_score(self, text: str, level: str = "full") -> dict:
        """
        Calculate various readability scores (Flesch Reading Ease, Flesch-Kincaid Grade Level,
        Gunning Fog Index) for the input text.
        """

        def get_scores(text_segment):
            text_segment = strip_markdown_markup(text_segment)
            if not text_segment or len(text_segment.split()) < 3:
                return {"flesch": None, "kincaid": None, "fog": None}
            return {
                "flesch": textstat.flesch_reading_ease(text_segment),
                "kincaid": textstat.flesch_kincaid_grade(text_segment),
                "fog": textstat.gunning_fog(text_segment),
            }

        if level == "full":
            return get_scores(text)

        elif level == "section":
            sections_data = parse_markdown_sections(text)
            result = {"full_text": get_scores(text), "sections": {}}

            for section_name, section_content in sections_data.items():
                if (
                    isinstance(section_content, str)
                    and section_name != "full_text"
                    and not section_name.endswith("_paragraphs")
                    and section_name != "paragraphs"
                ):
                    result["sections"][section_name] = get_scores(section_content)

            return result

        elif level == "paragraph":
            result = {"full_text": get_scores(text), "paragraphs": []}

            paragraphs = split_paragraphs(text)
            for i, paragraph in enumerate(paragraphs):
                result["paragraphs"].append(
                    {
                        "paragraph_number": i + 1,
                        "text": paragraph[:50] + "..." if len(paragraph) > 50 else paragraph,
                        "scores": get_scores(paragraph),
                    }
                )

            return result

        else:
            return {"error": f"Invalid level: {level}. Choose from 'full', 'section', or 'paragraph'."}

    def reading_time(self, text: str, level: str = "full") -> dict:
        """
        Estimate the reading time for the input text using textstat.
        """

        def get_reading_time(text_segment):
            if not text_segment:
                return 0
            # Convert milliseconds to minutes
            return textstat.reading_time(text_segment, ms_per_char=self.ms_per_character) / 60

        if level == "full":
            return {"full_text": get_reading_time(text)}

        elif level == "section":
            sections_data = parse_markdown_sections(text)
            result = {"full_text": get_reading_time(text), "sections": {}}

            for section_name, section_content in sections_data.items():
                if (
                    isinstance(section_content, str)
                    and section_name != "full_text"
                    and not section_name.endswith("_paragraphs")
                    and section_name != "paragraphs"
                ):
                    result["sections"][section_name] = get_reading_time(section_content)

            return result

        elif level == "paragraph":
            paragraphs = split_paragraphs(text)
            result = {"full_text": get_reading_time(text), "paragraphs": []}

            for i, paragraph in enumerate(paragraphs):
                result["paragraphs"].append(
                    {
                        "paragraph_number": i + 1,
                        "text": paragraph[:50] + "..." if len(paragraph) > 50 else paragraph,
                        "reading_time_minutes": get_reading_time(paragraph),
                    }
                )

            return result

        else:
            return {"error": f"Invalid level: {level}. Choose from 'full', 'section', or 'paragraph'."}
