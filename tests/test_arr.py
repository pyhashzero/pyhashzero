import pytest


def test_arr():
    import numpy as np
    from hz.arr import functional

    arr1 = [
        [[1., 2., 3.], [4., 5., 6.]],
        [[7., 8., 9.], [10., 11., 12.]]
    ]
    arr2 = [[2., 3., 4.], [5., 6., 7.]]

    print(np.std(np.asarray(arr1)))
    print(np.std(np.asarray(arr1), axis=0))
    print(np.std(np.asarray(arr1), axis=1))
    print(np.std(np.asarray(arr1), axis=2))
    print()
    print(functional.std(functional.asarray(arr1)))
    print(functional.std(functional.asarray(arr1), axis=0))
    print(functional.std(functional.asarray(arr1), axis=1))
    print(functional.std(functional.asarray(arr1), axis=2))
    print()

    print(np.asarray(arr1) - np.asarray(arr2))
    print(functional.asarray(arr1) - functional.asarray(arr2))


if __name__ == '__main__':
    pytest.main([__file__])
