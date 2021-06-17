import numpy as np
from utils.confusion_matrix import CM


def _verify_cm(trues, falses, expected_sum):
    res = [False for a in (trues + falses) if a != expected_sum]
    if len(res) > 0:
        raise ValueError(
            """
            At least one of the following conditions is violated:
               TP + FN = P
               TN + FP = N
            """
        )


class CMGenerator:

    def __init__(self, n_p, n_n, n_cm):
        """

        :param n_p:
        :param n_n:
        :param n_cm: number of splits
        """
        self.n_cm = n_cm
        self.n_p = n_p
        self.n_n = n_n
        self.all_cms = []

    def generate_cms_old(self):
        all_tp = np.linspace(0, self.n_p, self.n_cm, dtype=int)
        all_fn = self.n_p - all_tp
        _verify_cm(all_tp, all_fn, self.n_p)

        all_tn = np.linspace(0, self.n_n, self.n_cm, dtype=int)
        all_fp = self.n_n - all_tn
        _verify_cm(all_tn, all_fp, self.n_n)

        for i in range(len(all_tp)):
            for j in range(len(all_tn)):
                self.all_cms.append(
                    CM({'tp': all_tp[i],
                        'fn': all_fn[i],
                        'tn': all_tn[j],
                        'fp': all_fp[j]})
                )

    def generate_cms(self):
        all_tpr = np.linspace(0, 1.0, self.n_cm)
        all_tnr = np.linspace(0, 1.0, self.n_cm)
        for tpr in all_tpr:
            for tnr in all_tnr:
                self.all_cms.append(
                    CM({'tp': self.n_p * tpr,
                        'fn': self.n_p * (1 - tpr),
                        'tn': self.n_n * tnr,
                        'fp': self.n_n * (1 - tnr)})
                )

    def show_all_cms(self, limit: int = None):
        limit = len(self.all_cms) if limit is None else limit
        i = 0
        while i < limit:
            print('--[{}]-----------------------------------------'.format(i))
            print(self.all_cms[i])
            i += 1


if __name__ == "__main__":
    p, n = 2500, 2500
    gen = CMGenerator(n_p=p, n_n=n, n_cm=6)
    gen.generate_cms()
    print(len(gen.all_cms))
    for cm in gen.all_cms:
        print(cm)
    # n_ps = np.arange(100, 2501, 300)[::-1]
    # n_ns = np.asarray(5000 - n_ps)
    # cm_collection = []  # [[CM, ...], ...]
    # for n_p, n_n in zip(n_ps, n_ns):
    #     gen = CMGenerator(n_p=n_p, n_n=n_n, n_cm=12)
    #     gen.generate_cms()
    #     cm_collection.append(gen.all_cms)
    # print(len(cm_collection))
