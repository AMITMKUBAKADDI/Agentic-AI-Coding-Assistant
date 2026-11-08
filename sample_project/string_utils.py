"""String utilities module for text processing and validation."""

def reverse_string(text: str) -> str:
    """Return the reversed version of a given string."""
    return text[::-1]

def is_palindrome(text: str) -> bool:
    """Check if the given string is a palindrome, ignoring case and spaces."""
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]

def count_words(text: str) -> int:
    """Count the total number of words in a string."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())

def capitalize_words(text: str) -> str:
    """Capitalize the first letter of each word in the string."""
    return text.title()

