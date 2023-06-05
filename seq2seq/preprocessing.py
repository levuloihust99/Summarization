import unicodedata
from typing import Text


def unicode_normalize(text: Text):
    return unicodedata.normalize("NFKC", text)

class NFKCNormalizer(object):
    def __call__(self, text):
        return unicode_normalize(text)
