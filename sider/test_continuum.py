import numpy as np
from integrator import continuum_remove


def assert_allclose(a, b, tol=1e-6):
    a = np.asarray(a)
    b = np.asarray(b)
    assert np.all(np.abs(a - b) <= tol), f"Arrays differ:\n{a}\n{b}"


def test_flat_spectrum():
    s = [1.0, 1.0, 1.0, 1.0]
    cr = continuum_remove(s)
    # Continuum removal of a flat line should give ones
    assert_allclose(cr, [1.0, 1.0, 1.0, 1.0])


def test_v_shaped_absorption():
    s = [1.0, 0.5, 1.0]
    cr = continuum_remove(s)
    # The continuum should be [1,1,1], so cr equals original
    assert_allclose(cr, [1.0, 0.5, 1.0])


def test_rand_noise():
    rng = np.random.RandomState(0)
    s = (rng.rand(20) * 0.2 + 0.9).tolist()  # around 0.9-1.1
    cr = continuum_remove(s)
    # cr should be finite and positive
    assert np.all(np.isfinite(cr))
    assert np.all(np.array(cr) > 0)


if __name__ == '__main__':
    test_flat_spectrum()
    test_v_shaped_absorption()
    test_rand_noise()
    print('All continuum tests passed.')
