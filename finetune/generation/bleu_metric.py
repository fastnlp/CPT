import json
import warnings
import numpy as np
import nltk
from typing import List
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import itertools
from copy import deepcopy
import torch

class Ngrams(object):
    """
        Ngrams datastructure based on `set` or `list`
        depending in `exclusive`
    """

    def __init__(self, ngrams={}, exclusive=True):
        if exclusive:
            self._ngrams = set(ngrams)
        else:
            self._ngrams = list(ngrams)
        self.exclusive = exclusive

    def add(self, o):
        if self.exclusive:
            self._ngrams.add(o)
        else:
            self._ngrams.append(o)

    def __len__(self):
        return len(self._ngrams)

    def intersection(self, o):
        if self.exclusive:
            inter_set = self._ngrams.intersection(o._ngrams)
            return Ngrams(inter_set, exclusive=True)
        else:
            other_list = deepcopy(o._ngrams)
            inter_list = []

            for e in self._ngrams:
                try:
                    i = other_list.index(e)
                except ValueError:
                    continue
                other_list.pop(i)
                inter_list.append(e)
            return Ngrams(inter_list, exclusive=False)

    def union(self, *ngrams):
        if self.exclusive:
            union_set = self._ngrams
            for o in ngrams:
                union_set = union_set.union(o._ngrams)
            return Ngrams(union_set, exclusive=True)
        else:
            union_list = deepcopy(self._ngrams)
            for o in ngrams:
                union_list.extend(o._ngrams)
            return Ngrams(union_list, exclusive=False)

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0,len(sub)+1)] for _ in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]


class Metric(object):
    def __init__(self, toker):
        self.refs = []
        self.hyps = []
        self.toker = toker

    def forword(self, refs: List[List[str]], hyp: List[str]): # TODO: only applicable to token ids
        self.refs.append(refs)
        self.hyps.append(hyp)

    def calc_bleu_k(self, k):
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            bleu = corpus_bleu(self.refs, self.hyps, weights=weights,
                               smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            warnings.warn('the bleu is invalid')
            bleu = 0.
        return bleu

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen)-k):
                key = tuple(sen[i:i+k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        return dist

    def calc_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / len(ref)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores

    def calc_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / len(ref))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max !=0:
                score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores

    def _get_ngrams(self, n, text, exclusive=True):
        """Calcualtes n-grams.
        Args:
          n: which n-grams to calculate
          text: An array of tokens
        Returns:
          A set of n-grams
        """
        ngram_set = Ngrams(exclusive=exclusive)
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _get_word_ngrams(self, n, sentences, exclusive=True):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0

        if torch.distributed.get_rank() == 0:
            print(sentences)

        words = [x for y in sentences for x in y] # flatten the sentences
        if torch.distributed.get_rank() == 0:
            print("words", words)
        return self._get_ngrams(n, words, exclusive=exclusive)

    def f_r_p_rouge_n(self, evaluated_count, reference_count, overlapping_count):
        # Handle edge case. This isn't mathematically correct, but it's good enough
        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        return recall

    def calc_rouge_n(self, n=2, exclusive=True):
        """
        Computes ROUGE-N of two text collections of sentences.
        Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf
        Args:
          evaluated_sentences: The sentences that have been picked by the
                               summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram.  Defaults to 2.
        Returns:
          A tuple (f1, precision, recall) for ROUGE-N
        Raises:
          ValueError: raises exception if a param has len <= 0
        """
        if len(self.hyps) <= 0:
            raise ValueError("Hypothesis is empty.")
        if len(self.refs) <= 0:
            raise ValueError("Reference is empty.")

        evaluated_ngrams = self._get_word_ngrams(n, self.hyps, exclusive=exclusive)
        refs = [x[0] for x in self.refs]
        reference_ngrams = self._get_word_ngrams(n, refs, exclusive=exclusive)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        return self.f_r_p_rouge_n(evaluated_count, reference_count, overlapping_count)

    def close(self):
        result = {
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(3, 5)},
            **{f"bleu-{k}": 100 * self.calc_bleu_k(k) for k in range(4, 5)}
        }

        f1, scores = self.calc_unigram_f1()
        result['f1'] = 100 * f1
        result_list = {
            'f1': scores
        }

        rl, scores = self.calc_rouge_l()
        result['rouge-l'] = 100 * rl
        result_list.update({
            'rouge-l': scores
        })

        result["rouge-1"] = 100 * self.calc_rouge_n(n=1)
        result["rouge-2"] = 100 * self.calc_rouge_n(n=2)

        return result, result_list