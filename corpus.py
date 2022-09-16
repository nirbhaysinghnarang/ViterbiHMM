class Corpus:
    def __init__(self, train, test, val):
        self.train = train
        self.test = test
        self.val = val

        self.handleUnknowns(self.train['text'])
        self.dictionary = self.buildWordCountDictionary(self.train['text'])
        self.replaceUnknownWords(self.test['text'], self.dictionary)

    def replaceUnknownWords(self, setOfWords, vocabulary):
        for sentenceIdx in range(len(setOfWords)):
            for wordIdx in range(len(setOfWords[sentenceIdx])):
                if setOfWords[sentenceIdx][wordIdx] not in vocabulary:
                    setOfWords[sentenceIdx][wordIdx] = "<UNK>"

    def buildWordCountDictionary(self, corpus):
        wordDict = {}
        for sentence in corpus:
            for word in sentence:
                if word in wordDict:
                    wordDict[word] += 1
                else:
                    wordDict[word] = 1
        return wordDict

    def replaceInfrequentWordsWithUNK(self, corpus, dictionary):
        for (i, sentence) in enumerate(corpus):
            for (j, word) in enumerate(sentence):
                if dictionary[word] < 2:
                    corpus[i][j] = "<UNK>"
                    break

    def handleUnknowns(self, corpus):
        self.replaceInfrequentWordsWithUNK(
            corpus, self.buildWordCountDictionary(corpus))
