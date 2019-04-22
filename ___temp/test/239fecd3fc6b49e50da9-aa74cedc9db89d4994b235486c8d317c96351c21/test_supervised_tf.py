import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer
from supervised_tf import SupervisedTermWeights

from sklearn.utils.testing import assert_array_equal, assert_equal, assert_true


# Let's do language guessing.
docs = ["an apple a day keeps the doctor away",
        "time flies like an arrow",
        "the more the merrier",
        "the quick brown fox jumps over the lazy dog",
        "quod licet Iovi non licet bovi",
        "ut desint vires, tamen laudanda est voluntas",
        "gallia est omnis divisa in partes tres",
        "ceterum censeo carthaginem delendam esse",
        ]
y = ["en", "en", "en", "en", "la", "la", "la", "la"]

v = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
X = v.fit_transform(docs)


def test_supervised_term_weights():
    X_a = X.toarray()

    for weighting in SupervisedTermWeights._WEIGHTING:
        for reduction in SupervisedTermWeights._REDUCE:
            sup_tw = SupervisedTermWeights(weighting=weighting,
                                           reduce=reduction)
            X1 = sup_tw.fit_transform(X, y)
            X2 = sup_tw.fit(X, y).transform(X)

            assert_true(sp.isspmatrix(X1))
            assert_true(sp.isspmatrix(X2))
            assert_equal(X1.shape, X2.shape)

            X1_a = X1.toarray()
            assert_array_equal(X1_a, X_a * sup_tw.weights_)
