from unittest import TestCase


class TestLinearRegression(TestCase):
    def test_computeCost1(self):
        import numpy as num
        from source.linearregression import LinearRegression
        X = num.array([[1, 2], [1, 3], [1, 4], [1, 5]], num.float)
        y = num.array([[7], [6], [5], [4]], num.float)
        theta = num.array([[0.1], [0.2]], num.float)
        num.testing.assert_almost_equal(LinearRegression.computeCost(X, y, theta), 11.9450,  decimal=4)

    def test_computeCost2(self):
        import numpy as num
        from source.linearregression import LinearRegression
        X = num.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]], num.float)
        y = num.array([[7], [6], [5], [4]], num.float)
        theta = num.array([[0.1], [0.2], [0.3]], num.float)
        result = LinearRegression.computeCost(X, y, theta)
        num.testing.assert_almost_equal(result,  7.0175, decimal=4)

    def test_gradientDesent1(self):
        import numpy as num
        from source.linearregression import LinearRegression
        X = num.array([[1, 5], [1, 2], [1, 4], [1, 5]], num.float)
        y = num.array([[1], [6], [4], [2]], num.float)
        theta = num.array([[0.0], [0.0]], num.float)

        alpha = 0.01
        iterations = 1000
        result = num.array([[5.2148], [-0.5733]], num.float)

        num.testing.assert_array_almost_equal(LinearRegression.gradientDescent(X,y,theta,alpha,iterations), result, decimal=4)
