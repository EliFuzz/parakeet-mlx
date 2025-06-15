import re
import string
from dataclasses import dataclass
from typing import Optional, Set

from parakeet_mlx.audio.alignment import AlignedResult, AlignedSentence

# Default list of common filler words and speech disfluencies
DEFAULT_FILLER_WORDS = {
    # Common English filler words
    "um", "uh", "ah", "er", "eh", "m", "mm", "hmm", "mhm", "mm-hm", "mm-hmm", "mmm",
    "erm", "urm", "umm", "uhh", "ahh", "err", "emm", "yeah", "uh, yeah",
    
    # Hesitation sounds
    "h", "ooh", "oof", "phew", "tsk", "psh", "pfft", ".",
    
    # Throat clearing and breathing sounds
    "ahem", "cough", "sigh", "gasp", "sniff", "breath",
    
    # Partial words and false starts (common patterns)
    "i-", "the-", "and-", "but-", "so-", "well-", "like-", "you-", "mm-",
    
    # Repetitive confirmations
    "yeah yeah", "yes yes", "no no", "ok ok", "right right",
    
    # Common interjections that are often transcribed incorrectly
    "uh-huh", "mm-mm", "nuh-uh", "uh-oh", "oh-oh",
    
    # Single letter sounds (lowercase only, excluding meaningful words)
    "e", "o", "u", ""  # Only when they appear as standalone words
}

# Patterns for detecting filler-like content
FILLER_PATTERNS = [
    # Repeated characters (e.g., "uhhhhh", "mmmmm")
    re.compile(r'^([a-z])\1{2,}$', re.IGNORECASE),
    
    # Hyphenated incomplete words
    re.compile(r'^[a-z]+-$', re.IGNORECASE),
    
    # Very short repeated syllables
    re.compile(r'^([a-z]{1,2})\1+$', re.IGNORECASE),
    
    # Common breathing/throat sounds
    re.compile(r'^(ah+|eh+|oh+|uh+|mm+|hm+)$', re.IGNORECASE),
]


@dataclass
class FillerFilterConfig:
    """Configuration for filler word filtering"""
    enabled: bool = False
    confidence_threshold: float = 0.2
    custom_filler_words: Optional[Set[str]] = None
    filter_single_letters: bool = True
    filter_repeated_chars: bool = True
    filter_incomplete_words: bool = True
    min_word_length: int = 3  # Minimum length for words to be considered meaningful


class FillerWordFilter:
    def __init__(self, config: FillerFilterConfig):
        self.config = config
        self._filler_words = self._build_filler_word_set()
        self._patterns = FILLER_PATTERNS if config.filter_repeated_chars else []
        
    def _build_filler_word_set(self) -> Set[str]:
        """Build the complete set of filler words to filter"""
        filler_words = DEFAULT_FILLER_WORDS.copy()
        
        if self.config.custom_filler_words:
            filler_words.update(self.config.custom_filler_words)
        
        return {word.lower() for word in filler_words}
    
    def is_filler_word(self, text: str) -> bool:
        """
        Determine if a text segment is a filler word or disfluency

        Args:
            text: Text to check

        Returns:
            bool: True if the text should be filtered as a filler word
        """
        if not text or not self.config.enabled:
            return False

        normalized_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        original_text = text.strip()

        if len(normalized_text) < self.config.min_word_length and not self.config.filter_single_letters:
            return False

        if len(normalized_text) == 1:
            # "I" should never be filtered
            if original_text == "I":
                return False
            if normalized_text == "a":
                return False

        if normalized_text in self._filler_words:
            return True

        for pattern in self._patterns:
            if pattern.match(normalized_text):
                return True

        if self.config.filter_incomplete_words and normalized_text.endswith('-'):
            return True

        return False
    
    def filter_result(self, result: AlignedResult) -> AlignedResult:
        if not self.config.enabled or not result.text:
            return result
        
        filtered_sentences = []
        for sentence in result.sentences:
            filtered_sentence = self._filter_sentence(sentence)
            if filtered_sentence and filtered_sentence.text.strip():
                filtered_sentences.append(filtered_sentence)
        
        filtered_text = "".join(sentence.text for sentence in filtered_sentences)
        
        filtered_result = AlignedResult(
            text=filtered_text,
            sentences=filtered_sentences
        )
        
        return filtered_result
            
    
    def _filter_sentence(self, sentence: AlignedSentence) -> Optional[AlignedSentence]:
        filtered_tokens = []
        tokens = [sentence.tokens[i:i + 3] for i in range(0, len(sentence.tokens), 3)]
        
        for token in tokens:
            if not self.is_filler_word(''.join(item.text for item in token)):
                filtered_tokens.extend(token)
        
        if not filtered_tokens:
            return None
        
        filtered_text = "".join(token.text for token in filtered_tokens)
        
        return AlignedSentence(
            text=filtered_text,
            tokens=filtered_tokens
        )
    
    def filter_text_only(self, text: str) -> str:
        if not self.config.enabled or not text:
            return text
        
        words = text.split()
        filtered_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w\'-]', '', word).lower()
            
            if not self.is_filler_word(clean_word):
                filtered_words.append(word)
        
        return " ".join(filtered_words)
