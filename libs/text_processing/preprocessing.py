import unicodedata
from typing import Text
from io import StringIO


def _unicode_normalize(text: Text):
    return unicodedata.normalize("NFKC", text)


def _is_punctuation(ch):
    """Checks whether `ch` is a punctuation character."""
    codepoint = ord(ch)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (33 <= codepoint <= 47) or
        (58 <= codepoint <= 64) or
        (91 <= codepoint <= 96) or
        (123 <= codepoint <= 126)
    ):
        return True

    category = unicodedata.category(ch)

    if category.startswith("P"):
        return True

    return False


def _is_control(ch):
    """Checks whether `ch` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if ch in ("\t", "\n", "\v", "\f", "\r",
        "\u001C", # File Separator
        "\u001D", # Group Separator
        "\u001E", # Information Separator Two
        "\u001F", # Information Separator One
        "\u0085", # Next Line
    ):
        return False

    category = unicodedata.category(ch)

    if category in ("Cc", "Cf"):
        return True

    return False


def _is_whitespace(ch):
    """Checks whether `ch` is a whitespace character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if ch in ("\t", "\n", "\v", "\f", "\r",
        "\u001C", # File Separator
        "\u001D", # Group Separator
        "\u001E", # Information Separator Two
        "\u001F", # Information Separator One
        "\u0085", # Next Line
    ):
        return True

    category = unicodedata.category(ch)

    if category == "Zl" or category == "Zp": # Line Separator and Paragraph Separator
        return True

    if category == "Zs" and ch != "\u1680": # \u1680 is Ogham Space Mark
        return True
    
    return False


def _is_chinese_char(ch):
    """Checks whether `ch` is a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(ch) # codepoint of character `ch`
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for ch in text:
        if _is_chinese_char(ch):
            if not (output and output[-1] == " "):
                output.append(" ")
            output.append(ch)
            output.append(" ")
        else:
            output.append(ch)
    return "".join(output)


def clean_text(text: Text, tokenize_chinese_chars: bool = False, replace_newline: bool = False):
    text = _unicode_normalize(text)
    output = []
    for ch in text:
        codepoint = ord(ch)
        if codepoint == 0xfffc or codepoint == 0xfffd or _is_control(ch):
            continue
        if _is_whitespace(ch):
            if ch == "\n" and not replace_newline:
                output.append(ch)
            else:
                output.append(" ")
        else:
            output.append(ch)
    output_text = "".join(output)
    if tokenize_chinese_chars:
        output_text = _tokenize_chinese_chars(output_text)
    return output_text


def strip_lines(text: Text):
    stream = StringIO(text)
    output = []
    for line in stream:
        line = line.strip()
        if line == "":
            continue
        output.append(line)
    return "\n".join(output) + "\n"


def whitespace_tokenize(text: Text):
    """Run basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens