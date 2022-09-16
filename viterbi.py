from tracemalloc import start


class Viterbi:
    def __init__(self, hmm, observation):
        self.states = hmm.states
        self.start_state_matrix = hmm.start_state_matrix
        self.emission_matrix = hmm.emission_matrix
        self.transition_matrix = hmm.transition_matrix
        self.observation = observation

    def populateProbabilities(self):
       # create V
        V = [{}]
        # initialize V
        for (i, tag) in enumerate(self.states):
            V[0][tag] = {
                "prob": self.start_state_matrix[tag] * self.emission_matrix[(tag, self.observation[0])],
                "prev": None
            }
        for t in range(1, len(self.observation)):
            V.append({})
            for tag in self.states:
                max_trans_prob = V[t - 1][self.states[0]]["prob"] * \
                    self.transition_matrix[(self.states[0], tag)]
                prev_tag_selected = self.states[0]
                for prev_tag in self.states[1:]:
                    tr_prob = V[t - 1][prev_tag]["prob"] * \
                        self.transition_matrix[(prev_tag, tag)]
                    if tr_prob > max_trans_prob:
                        max_trans_prob = tr_prob
                        prev_tag_selected = prev_tag
                max_prob = max_trans_prob * \
                    self.emission_matrix[(tag, self.observation[t])]
                V[t][tag] = {"prob": max_prob, "prev": prev_tag_selected}
        return V

    def backTrack(self, V):
        opt = []
        max_prob = -1
        best_st = None
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]
        for i in range(len(self.observation)):
            if(self.observation[i][0].islower()):
                opt[i] = "O"
        return opt

    def run(self):
        return self.backTrack(self.populateProbabilities())
