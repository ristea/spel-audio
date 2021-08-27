from sklearn.metrics import label_ranking_average_precision_score
import numpy as np


class StatsManager:

    def __init__(self, config):
        self.config = config

    def get_stats(self, predictions, labels):
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)

        lravp = []
        for i in range(0, len(labels)):
            if sum(labels[i]) == 0:
                continue

            lravp.append(label_ranking_average_precision_score(np.expand_dims(labels[i], 0),
                                                               np.expand_dims(predictions[i], 0)))

        lravp = np.mean(lravp)
        return lravp
