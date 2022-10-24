import Dao_email
import linear_logistic
import naver_extraction
import matplotlib.pyplot as plt
import numpy as np


def test():
    train_set, test_set = naver_extraction.making_doclist(0.8, Dao_email.connection_sqlite)

    v = linear_logistic.making_BOV(train_set, linear_logistic.split())

    tdm, target = linear_logistic.separate_target(train_set, linear_logistic.split(), v)

    test_tdm, test_target = linear_logistic.separate_target(test_set, linear_logistic.split(), v)

    theta, history = linear_logistic.logistic_regression(tdm, target, 1000, 0.1)

    l_theta, l_history = linear_logistic.linear_regression(tdm, target, 1000, 0.1)

    plt.plot(history)
    plt.plot(l_history, c='red')
    plt.show()

    print("++++++++++linear_regression++++++++++")
    l_result = linear_logistic.scoreing(test_tdm, l_theta, test_target, 1)

    print("++++++++++logistic_regression++++++++++")
    result = linear_logistic.scoreing(test_tdm, theta, test_target)

    return np.array(l_result), np.array(result)


def make_average(cnt):
    l_result = np.array([0] * 4)
    result = np.array([0] * 4)

    for _ in range(cnt):
        a, b = test()
        l_result = l_result + a
        result = result + b

    l_total = l_result / cnt
    total = result / cnt

    print("")

    print("====================Total======================")

    print("")

    print("++++++++++linear_regression++++++++++")
    print("precision:" + str(l_total[0]))
    print("accuracy:" + str(l_total[1]))
    print("Recall:" + str(l_total[2]))
    print("F1-score" + str(l_total[3]))

    print("===========================================")

    print("++++++++++logistic_regression++++++++++")
    print("precision:" + str(total[0]))
    print("accuracy:" + str(total[1]))
    print("Recall:" + str(total[2]))
    print("F1-score" + str(total[3]))
