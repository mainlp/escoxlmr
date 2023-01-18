import numpy as np


class MRR():
    def __init__(self):
        super(MRR, self).__init__()
        self.rr = []
        self.str = "mrr"

    def score(self, preds, golds, mask, vocabulary):
        y_pred = preds.detach().cpu().numpy()
        y_true = golds.detach().cpu().numpy()

        for i in range(len(y_pred)):
            y_t = np.where(y_true[i] == 1)[0]
            y_p_index = np.flip(np.argsort(y_pred[i]), 0)
            for i in range(len(y_p_index)):
                if y_p_index[i] in y_t:
                    self.rr.append(1 / float(i + 1))
                    break

    def reset(self):
        self.rr = []

    def get_score(self):
        return self.str, float(np.mean(np.array(self.rr)))
