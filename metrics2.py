import torch
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer


def bleu_score_fn(method_no: int = 4, ref_type='corpus'):
    """
    :param method_no:
    :param ref_type: 'corpus' or 'sentence'
    :return: bleu score
    """
    smoothing_method = getattr(SmoothingFunction(), f'method{method_no}')

    def bleu_score_corpus(reference_corpus: list, candidate_corpus: list, n: int = 4):
        """
        :param reference_corpus: [b, 5, var_len]
        :param candidate_corpus: [b, var_len]
        :param n: size of n-gram
        """
        weights = [1 / n] * n
        return corpus_bleu(reference_corpus, candidate_corpus,
                           smoothing_function=smoothing_method, weights=weights)

    def bleu_score_sentence(reference_sentences: list, candidate_sentence: list, n: int = 4):
        """
        :param reference_sentences: [5, var_len]
        :param candidate_sentence: [var_len]
        :param n: size of n-gram
        """
        weights = [1 / n] * n
        return sentence_bleu(reference_sentences, candidate_sentence,
                             smoothing_function=smoothing_method, weights=weights)

    if ref_type == 'corpus':
        return bleu_score_corpus
    elif ref_type == 'sentence':
        return bleu_score_sentence

def rouge_score_fn():
    """
    Create a ROUGE scorer function.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    def rouge_score(reference_corpus: list, candidate_corpus: list):
        """
        Calculate ROUGE scores for each pair of reference and candidate sentences.
        :param reference_corpus: list of lists of reference sentences [[[ref1a, ref1b, ...], ...]]
        :param candidate_corpus: list of candidate sentences [[cand1], [cand2], ...]
        :return: dict of average ROUGE scores
        """
        scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeLsum': 0.0}
        count = 0
        
        for references, candidate in zip(reference_corpus, candidate_corpus):
            # Flatten the list of reference sentences for each candidate
            references = [' '.join(ref) for ref in references]
            candidate = ' '.join(candidate[0])  # Assuming each candidate is a list of words
            score = scorer.score(' '.join(references), candidate)
            scores['rouge1'] += score['rouge1'].fmeasure
            scores['rouge2'] += score['rouge2'].fmeasure
            scores['rougeLsum'] += score['rougeLsum'].fmeasure
            count += 1

        # Average the scores
        for key in scores:
            scores[key] /= count
        return scores

    return rouge_score


def accuracy_fn(ignore_value: int = 0):
    def accuracy_ignoring_value(source: torch.Tensor, target: torch.Tensor):
        mask = target != ignore_value
        return (source[mask] == target[mask]).sum().item() / mask.sum().item()

    return accuracy_ignoring_value
