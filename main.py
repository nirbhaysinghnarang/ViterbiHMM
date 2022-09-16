
from data import Data
from hmm import HMM
from viterbi import Viterbi
from corpus import Corpus

TAGS = ["B-ORG", "I-ORG", "B-PER", "I-PER",
        "B-LOC", "I-LOC", "B-MISC", "I-MISC", "O"]


dataset = Data('train.json', 'test.json', 'val.json')
corpus = Corpus(dataset.train, dataset.test, dataset.val)
hmm = HMM(corpus.train['text'], corpus.train['NER'], 1, states=TAGS)
for sentence in corpus.test['text']:
    viterbi = Viterbi(hmm, sentence)
    print(f"{' '.join(sentence)} -> {' '.join(viterbi.run())}")
