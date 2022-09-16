import itertools


class HMM():
    def __init__(self, tokens, labels, smoothing_factor, states):
        self.states = states
        self.transition_matrix = self.build_transition_matrix(
            labels=labels, k=smoothing_factor)
        self.emission_matrix = self.build_emission_matrix(
            tokens=tokens, labels=labels, k=smoothing_factor)
        self.start_state_matrix = self.build_start_state_matrix(
            labels=labels, k=smoothing_factor)

    def buildTagBigrams(self):
        TAG_BIGRAMS = list(itertools.permutations(self.states, 2))
        for tag in self.states:
            TAG_BIGRAMS.append((tag, tag))
        return TAG_BIGRAMS

    def init_trans_matrix(self):
        trans_matrix = {}
        for bigram in self.buildTagBigrams():
            trans_matrix[bigram] = 0
        return trans_matrix

    def init_emission_matrix(self, tokens):
        emission_matrix = {}
        for token_seq in tokens:
            for token in token_seq:
                for tag in self.states:
                    emission_matrix[(tag, token)] = 0
        return emission_matrix

    def init_start_state_matrix(self):
        start_matrix = {}
        for tag in self.states:
            start_matrix[tag] = 0
        return start_matrix

    def normalize_trans_matrix(self, matrix, totalMatrix, k):
        return {key:
                (matrix[key]+k) / (totalMatrix[key[0]])
                for key in matrix
                }

    def normalize_emit_matrix(self, matrix, totalMatrix, k):
        return {key:
                (matrix[key]+k) / (totalMatrix[key[0]])
                for key in matrix
                }

    def build_transition_matrix(self, labels, k=0):
        transition_matrix = self.init_trans_matrix()
        label_preceding_counts = {}
        for tag in self.states:
            label_preceding_counts[tag] = 0
        count = 0
        for label_sequence in labels:
            for (i, label) in enumerate(label_sequence):
                count += 1
                if (i != len(label_sequence)-1):
                    label_preceding_counts[label] += 1
                    label_tuple = (label, label_sequence[i+1])
                    transition_matrix[label_tuple] += 1
        return self.normalize_trans_matrix(
            transition_matrix, label_preceding_counts, k)

    def build_emission_matrix(self, tokens, labels, k=0):
        emission_matrix = self.init_emission_matrix(tokens)
        tag_count_dict = {}
        for tag in self.states:
            tag_count_dict[tag] = 0
        for (i, token_seq) in enumerate(tokens):
            for (j, token) in enumerate(token_seq):
                tag_count_dict[labels[i][j]] += 1
                emission_tuple = (labels[i][j], tokens[i][j])
                emission_matrix[emission_tuple] += 1
        return self.normalize_emit_matrix(
            emission_matrix, tag_count_dict, k)

    def build_start_state_matrix(self, labels, k=0):
        start_matrix = self.init_start_state_matrix()
        total = 0
        for label_seq in labels:
            for label in label_seq:
                total += 1
                start_matrix[label] += 1
        for key in start_matrix:
            start_matrix[key] = ((start_matrix[key]+k)/total)
        return start_matrix
