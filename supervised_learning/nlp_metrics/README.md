# NLP metrics

## BLEU score

BLEU (Bilingual Evaluation Understudy) is a metric for evaluating a generated sentence to a reference sentence. It is based on the precision of n-grams in the generated sentence compared to the reference sentence. BLEU score is a number between 0 and 1, where 1 means the generated sentence is identical to the reference sentence.

## TASKS

| Task                                                   | Description                                                |
|--------------------------------------------------------|------------------------------------------------------------|
| [Unigram BLEU score](./0-uni_bleu.py)                  | Calculates the unigram BLEU score for a sentence           |
| [n-gram BLEU score](./1-ngram_bleu.py)                 | Calculates the n-gram BLEU score for a sentence            |
| [Cumulative n-gram BLEU score](./2-cumulative_bleu.py) | Calculates the cumulative n-gram BLEU score for a sentence |

