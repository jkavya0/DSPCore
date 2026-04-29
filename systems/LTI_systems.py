import numpy as np


# -----------------------
# SYSTEM DEFINITION
# y[n] = x[n] + 2x[n-1]
# -----------------------
def system(x):
    y = np.zeros(len(x))

    for n in range(len(x)):
        y[n] = x[n]

        if n - 1 >= 0:
            y[n] += 2 * x[n - 1]

    return y


# -----------------------
# TRUE SHIFT (ZERO PADDING)
# -----------------------
def shift_signal(x, shift):
    y = np.zeros(len(x))

    # shift right (delay)
    if shift > 0:
        for i in range(len(x) - shift):
            y[i + shift] = x[i]

    # shift left (advance)
    elif shift < 0:
        for i in range(-shift, len(x)):
            y[i + shift] = x[i]

    # no shift
    else:
        y = x.copy()

    return y


# -----------------------
# TEST LINEARITY
# -----------------------
def test_linearity():
    x1 = np.array([1, 2, 0, 0])
    x2 = np.array([0, 1, 1, 0])

    a = 2
    b = 3

    left = system(a * x1 + b * x2)
    right = a * system(x1) + b * system(x2)

    print("Linearity Test:", np.allclose(left, right))


# -----------------------
# TEST TIME INVARIANCE
# -----------------------
def test_time_invariance():
    x = np.array([1, 2, 3, 0, 0])

    shift = 2

    # Step 1: shift input
    x_shifted = shift_signal(x, shift)

    # Step 2: system output
    y1 = system(x)

    # Step 3: system output of shifted input
    y2 = system(x_shifted)

    # Step 4: shift original output
    y1_shifted = shift_signal(y1, shift)

    print("Time Invariance Test:", np.allclose(y2, y1_shifted))


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    test_linearity()
    test_time_invariance()