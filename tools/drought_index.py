import numpy as np
import scipy.stats as st


class drought_index_clf:
    # Classifier based on drought indices.
    def __init__(self, threshold=-1.645, scoring="f1_score", search_thresholds=None):
        assert scoring in ("f1_score", "balanced_accuracy_score")
        self._threshold = threshold
        self.scoring = scoring
        self._fit_value = None

        if search_thresholds is None:
            self.search_threshold = np.linspace(start=-3., stop=0., num=24, endpoint=False, dtype=np.float32)
        else:
            self.search_threshold = search_thresholds

    def fit(self, x, y):
        best_results = [0., 0.]
        for th in self.search_threshold:
            y_pred = self.idx2prob(data=x, mu=th)
            y_pred = (y_pred > 0.5).astype(np.float32)
            value = eval(self.scoring)(y, y_pred)
            if value > best_results[0]:
                best_results = [value, th]

        self._fit_value = best_results[0]
        self._threshold = best_results[1]

    def predict(self, x):
        y_pred = self.predict_prob(x)
        y_pred = (y_pred > 0.5).astype(np.float32)
        return y_pred

    def predict_prob(self, x):
        y_pred_prob = self.idx2prob(data=x, mu=self._threshold)
        return y_pred_prob

    @staticmethod
    def idx2prob(data, mu=0., sigma=1.):
        if isinstance(data, np.ndarray):
            prob = st.norm.cdf(x=mu, loc=data, scale=sigma)
        else:
            prob = st.norm.cdf(x=mu, loc=np.array([data]), scale=sigma)
        return prob.astype(np.float32)


drought_idx = drought_index_clf
DI = drought_index_clf
DROUGHT_INDEX = drought_index_clf

