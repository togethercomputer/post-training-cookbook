from post_training_cookbook.datasets.math_grader import boxed_math_reward, math_equal


class TestMathEqual:
    def test_dfrac_matches_frac(self):
        assert math_equal("\\dfrac{1}{2}", "\\frac{1}{2}")

    def test_frac_matches_decimal(self):
        assert math_equal("\\frac{1}{2}", "0.5")

    def test_degrees_stripped(self):
        assert math_equal("90", "90^\\circ")

    def test_reordered_sum(self):
        assert math_equal("4 + 4\\sqrt{2}", "4\\sqrt{2} + 4")

    def test_text_wrapper(self):
        assert math_equal("October 30", "\\text{October 30}")

    def test_pi_fraction(self):
        assert math_equal("\\dfrac{5\\pi}{3}", "\\frac{5}{3}\\pi")

    def test_infinity(self):
        assert math_equal("\\infty", "\\infty")

    def test_distinct_values_not_equal(self):
        assert not math_equal("\\frac{1}{2}", "\\frac{1}{3}")

    def test_distinct_integers_not_equal(self):
        assert not math_equal("42", "43")

    def test_nth_root(self):
        assert math_equal("\\sqrt[3]{8}", "2")

    def test_polynomial_identity(self):
        assert math_equal("x^2 + 2x + 1", "(x+1)^2")

    def test_set_order_invariant(self):
        assert math_equal("\\{1,2,3\\}", "\\{3,2,1\\}")

    def test_unparseable_falls_through_to_string(self):
        assert not math_equal("garbagexyz", "\\frac{1}{2}")

    def test_distinct_text_not_equal(self):
        assert not math_equal("October 30", "October 31")


class TestBoxedMathReward:
    def test_correct_boxed(self):
        assert boxed_math_reward("so the answer is \\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}") == 1.0

    def test_wrong_boxed(self):
        assert boxed_math_reward("\\boxed{7}", "8") == 0.0

    def test_no_box(self):
        assert boxed_math_reward("the answer is 1/2", "\\frac{1}{2}") == 0.0

    def test_empty_target(self):
        assert boxed_math_reward("\\boxed{5}", "") == 0.0
