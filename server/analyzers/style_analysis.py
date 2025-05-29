"""Style analysis functionality."""


class StyleAnalyzer:
    """Handles style-related text analysis."""

    def __init__(self, nlp_model):
        self.nlp = nlp_model

    def passive_voice_detection(self, text: str) -> list[str]:
        """
        Detect sentences potentially written in passive voice using a simplified rule-based approach.

        Looks for patterns like auxiliary verb + past participle (e.g., "was written").
        Note: This is a basic detection and might not catch all passive constructions or might have false positives.
        """
        doc = self.nlp(text)
        passive_sentences = []

        for sent in doc.sents:
            # Simple passive voice detection: aux verb + past participle
            # This is a simplified approach and might miss some complex passive constructions
            for token in sent:
                if token.dep_ == "auxpass" or (
                    token.pos_ == "AUX" and any(child.tag_ == "VBN" for child in token.children)
                ):
                    passive_sentences.append(sent.text)
                    break

        return passive_sentences
