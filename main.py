import Dao_email
import linear_logistic
import naver_extraction


def test():
    train_set, test_set = naver_extraction.making_doclist(0.8, Dao_email.connection_sqlite)

    v = linear_logistic.making_BOV(train_set,linear_logistic.split())

    tdm, target = linear_logistic.separate_target(train_set, linear_logistic.split(), v)

    test_tdm, test_target = linear_logistic.separate_target(test_set, linear_logistic.split(), v)

    theta, history = linear_logistic.logistic_regression(tdm, target, 1000, 0.1)

    linear_logistic.scoreing(test_tdm, theta, test_target)


