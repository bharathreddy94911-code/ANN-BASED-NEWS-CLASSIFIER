import re
from typing import List

# Regular expression tokenizer for alphanumeric words
TOKEN_REGEX = re.compile(r'\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)?\b')

def clean_text(text: str) -> str:
    """Clean text by unescaping HTML tags, removing backslashes, extra spaces."""
    if not text:
        return ""
    # Remove HTML entities/tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Replace backslashes (frequently present in AG News like \b)
    text = text.replace('\\', ' ')
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def tokenize(text: str) -> List[str]:
    """Tokenize lowercased, cleaned text into a list of words."""
    cleaned = clean_text(text)
    return TOKEN_REGEX.findall(cleaned)
