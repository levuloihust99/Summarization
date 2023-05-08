import itertools
import collections
from typing import List, Tuple, Counter, Dict, Union, Callable

NGramsType = Counter[Tuple[str]]
ScoreType = Dict[str, float]
RougeType = Dict[str, Dict[str, float]]

try:
    from math import isclose
except ImportError:
    def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        # type: (float, float, float, float) -> bool
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def _get_weight_func(weight, inverse):
    # type: (float, bool) -> Callable[[float], float]
    if weight < 1:
        raise ValueError('Invalid weight {}: expected >= 1'.format(weight))

    if inverse:
        weight = 1 / weight

    return lambda x: x ** weight


def _format_score(fscore, precision, recall):
    # type: (float, float, float) -> Dict[str, float]
    return {'r': recall, 'p': precision, 'f': fscore}


def _f_score(precision, recall, alpha):
    # type: (float, float, float) -> float
    if not 0 <= alpha <= 1:
        raise ValueError(
            'Invalid alpha {}: expected between [0, 1]'.format(alpha))

    if isclose(precision, 0) or isclose(recall, 0):
        return 0.0

    return recall * precision / (alpha * recall + (1 - alpha) * precision)


def _div_or_zero(dividend, divisor):
    # type: (float, float) -> float
    if isclose(divisor, 0):
        return 0.0
    else:
        return dividend / divisor


def _f_p_r_score(match_score, hyp_score, ref_score, alpha):
    # type: (float, float, float, float) -> Dict[str, float]
    precision = _div_or_zero(match_score, hyp_score)
    recall = _div_or_zero(match_score, ref_score)
    fscore = _f_score(precision, recall, alpha)
    return _format_score(fscore, precision, recall)


def _flatten(sentences):
    # type: (List[List[str]]) -> List[str]
    return list(itertools.chain.from_iterable(sentences))


class _Match(collections.namedtuple('BaseMatch', 'matches hyp_size ref_size')):
    def __add__(self, other):
        # type: (Union[_Match, int]) -> _Match
        if isinstance(other, int) and other == 0:
            return self
        elif isinstance(other, _Match):
            return _Match(self.matches + other.matches,
                          self.hyp_size + other.hyp_size,
                          self.ref_size + other.ref_size)
        else:
            raise ValueError('Unexpected addend {}'.format(other))

    def __radd__(self, other):
        # type: (Union[_Match, int]) -> _Match
        return self.__add__(other)

    def to_score(self, alpha):
        # type: (float) -> Dict[str, float]
        return _f_p_r_score(self.matches, self.hyp_size, self.ref_size, alpha)

    def to_weighted_score(self, alpha, weight):
        # type: (float, float) -> Dict[str, float]
        inv_weight_func = _get_weight_func(weight, inverse=True)
        precision = inv_weight_func(_div_or_zero(self.matches, self.hyp_size))
        recall = inv_weight_func(_div_or_zero(self.matches, self.ref_size))
        fscore = _f_score(precision, recall, alpha)
        return _format_score(fscore, precision, recall)


def _build_ngrams(sent, n):
    # type: (List[str], int) -> NGramsType
    ngrams = collections.Counter()
    for i in range(len(sent) - n + 1):
        ngrams[tuple(sent[i:i + n])] += 1
    return ngrams


def _count_ngrams(ngrams):
    # type: (NGramsType) -> int
    return sum(ngrams.values())


def _intersect_ngrams(hyp_ngrams, ref_ngrams):
    # type: (NGramsType, NGramsType) -> NGramsType
    return hyp_ngrams & ref_ngrams


def _union_ngrams(ngrams, other):
    # type: (NGramsType, NGramsType) -> NGramsType
    return ngrams | other


def _rouge_n_sentence_level(hyp, ref, n):
    # type: (List[str], List[str], int) -> _Match
    hyp_ngrams = _build_ngrams(hyp, n)
    ref_ngrams = _build_ngrams(ref, n)
    match_ngrams = _intersect_ngrams(hyp_ngrams, ref_ngrams)
    return _Match(_count_ngrams(match_ngrams), _count_ngrams(hyp_ngrams),
                  _count_ngrams(ref_ngrams))
