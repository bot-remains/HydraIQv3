import spacy

# Load spaCy's model for semantic similarity
nlp = spacy.load("en_core_web_md")

# Groundwater-related phrases
groundwater_phrases = [
    "groundwater",
    "aquifer",
    "water table",
    "percolation",
    "wells",
    "subsurface water",
    "hydrology",
]

# Function to check relevance
def is_related_to_groundwater(question, threshold=0.7):
    question_doc = nlp(question)
    for phrase in groundwater_phrases:
        phrase_doc = nlp(phrase)
        if question_doc.similarity(phrase_doc) > threshold:
            return True
    return False
