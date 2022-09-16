
from data import Data
from hmm import HMM
from viterbi import Viterbi
from corpus import Corpus
from submit import create_submission
TAGS = ["B-ORG", "I-ORG", "B-PER", "I-PER",
        "B-LOC", "I-LOC", "B-MISC", "I-MISC", "O"]


dataset = Data('train.json', 'test.json', 'val.json')
corpus = Corpus(dataset.train, dataset.test, dataset.val)


test_idx_flat = [y for x in dataset.test['index'] for y in x]
output = []

hmm = HMM(corpus.train['text'], corpus.train['NER'], 1, states=TAGS)
for sentence in corpus.test['text']:
    output.append(Viterbi(hmm, sentence).run())
output_flat = [y for x in output for y in x]
create_submission("milestone.csv", output_flat, test_idx_flat)
