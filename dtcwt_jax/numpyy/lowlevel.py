from __future__ import absolute_import, division
from jax import lax, random, numpy as jnp

__all__ = [
    "colfilter",
    "colifilt",
    "coldfilt",
]

import numpy as np

# from six.moves import range
from dtcwt_jax.utils import (
    as_column_vector,
    asfarray,
    appropriate_complex_type_for,
    reflect,
)


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    # (Shamelessly cribbed from scipy.)
    newsize = jnp.array(newsize)
    currsize = jnp.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


# This is to allow easy replacement of these later with, possibly, GPU versions
_rfft = jnp.fft.rfft
_irfft = jnp.fft.irfft


def _column_convolve(X, h):
    """Convolve the columns of *X* with *h* returning only the 'valid' section,
    i.e. those values unaffected by zero padding. Irrespective of the ftype of
    *h*, the output will have the dtype of *X* appropriately expanded to a
    floating point type if necessary.

    We assume that h is small and so direct convolution is the most efficient.

    """
    Xshape = jnp.array(X.shape)
    h = h.flatten().astype(X.dtype)
    h_size = h.shape[0]

    full_size = X.shape[0] + h_size - 1
    Xshape=Xshape.at[0].set(full_size)

    out = jnp.zeros(Xshape, dtype=X.dtype)
    for idx in range(h_size):
        # out[idx : (idx + X.shape[0]), ...] += X * h[idx]
        out=out.at[idx : (idx + X.shape[0]), ...].set(out[idx : (idx + X.shape[0]), ...] + X * h[idx]) 

    outShape = Xshape.copy()
    outShape=outShape.at[0].set(abs(X.shape[0] - h_size) + 1)
    return _centered(out, outShape)


def colfilter(X, h):
    """Filter the columns of image *X* using filter vector *h*, without decimation.
    If len(h) is odd, each output sample is aligned with each ijnput sample
    and *Y* is the same size as *X*.  If len(h) is even, each output sample is
    aligned with the mid point of each pair of ijnput samples, and Y.shape =
    X.shape + [1 0].

    :param X: an image whose columns are to be filtered
    :param h: the filter coefficients.
    :returns Y: the filtered image.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """

    # Interpret all ijnputs as arrays
    X = asfarray(X)
    h = as_column_vector(h)

    r, c = X.shape
    m = h.shape[0]
    m2 = jnp.fix(m * 0.5)


    # Symmetrically extend with repeat of end samples.
    # Use 'reflect' so r < m2 works OK.
    xe = reflect(jnp.arange(-m2, r + m2, dtype=int), -0.5, r - 0.5)

    # Perform filtering on the columns of the extended matrix X(xe,:), keeping
    # only the 'valid' output samples, so Y is the same size as X if m is odd.
    Y = _column_convolve(X[xe, :], h)

    return Y


def coldfilt(X, ha, hb):
    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e. :math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext        top edge                     bottom edge       ext
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a

    The output is decimated by two from the input sample rate and the results
    from the two filters, Ya and Yb, are interleaved to give Y.  Symmetric
    extension with repeated end samples is used on the composite X columns
    before each filter is applied.

    Raises ValueError if the number of rows in X is not a multiple of 4, the
    length of ha does not match hb or the lengths of ha or hb are non-even.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 4 != 0:
        raise ValueError("No. of rows in X must be a multiple of 4")

    if ha.shape != hb.shape:
        raise ValueError("Shapes of ha and hb must be the same")

    if ha.shape[0] % 2 != 0:
        raise ValueError("Lengths of ha and hb must be even")

    m = ha.shape[0]
    m2 = jnp.fix(m * 0.5)

    # Set up vector for symmetric extension of X with repeated end samples.
    xe = reflect(jnp.arange(-m, r + m), -0.5, r - 0.5)

    # Select odd and even samples from ha and hb. Note that due to 0-indexing
    # 'odd' and 'even' are not perhaps what you might expect them to be.
    hao = as_column_vector(ha[0:m:2])
    hae = as_column_vector(ha[1:m:2])
    hbo = as_column_vector(hb[0:m:2])
    hbe = as_column_vector(hb[1:m:2])
    t = jnp.arange(5, r + 2 * m - 2, 4)
    r2 = r // 2
    Y = jnp.zeros((r2, c), dtype=X.dtype)

    if jnp.sum(ha * hb) > 0:
        s1 = slice(0, r2, 2)
        s2 = slice(1, r2, 2)
    else:
        s2 = slice(0, r2, 2)
        s1 = slice(1, r2, 2)

    # Perform filtering on columns of extended matrix X(xe,:) in 4 ways.
    Y=Y.at[s1, :].set(_column_convolve(X[xe[t - 1], :], hao) + _column_convolve(
        X[xe[t - 3], :], hae
    ))
    Y=Y.at[s2, :].set(_column_convolve(X[xe[t], :], hbo) + _column_convolve(
        X[xe[t - 2], :], hbe
    ))

    return Y


def colifilt(X, ha, hb):
    """Filter the columns of image X using the two filters ha and hb =
    reverse(ha).  ha operates on the odd samples of X and hb on the even
    samples.  Both filters should be even length, and h should be approx linear
    phase with a quarter sample advance from its mid pt (i.e `:math:`|h(m/2)| >
    |h(m/2 + 1)|`).

    .. code-block:: text

                          ext       left edge                      right edge       ext
        Level 2:        !               |               !               |               !
        +q filt on x      b       b       a       a       a       a       b       b
        -q filt on o          a       a       b       b       b       b       a       a
        Level 1:        !               |               !               |               !
        odd filt on .    b   b   b   b   a   a   a   a   a   a   a   a   b   b   b   b
        odd filt on .      a   a   a   a   b   b   b   b   b   b   b   b   a   a   a   a

    The output is interpolated by two from the input sample rate and the
    results from the two filters, Ya and Yb, are interleaved to give Y.
    Symmetric extension with repeated end samples is used on the composite X
    columns before each filter is applied.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
    .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
    .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000

    """
    # Make sure all inputs are arrays
    X = asfarray(X)
    ha = asfarray(ha)
    hb = asfarray(hb)

    r, c = X.shape
    if r % 2 != 0:
        raise ValueError("No. of rows in X must be a multiple of 2")

    if ha.shape != hb.shape:
        raise ValueError("Shapes of ha and hb must be the same")

    if ha.shape[0] % 2 != 0:
        raise ValueError("Lengths of ha and hb must be even")

    m = ha.shape[0]
    m2 = jnp.fix(m * 0.5)

    Y = jnp.zeros((r * 2, c), dtype=X.dtype)
    if not jnp.any(jnp.nonzero(X[:])[0]):
        return Y

    if m2 % 2 == 0:
        # m/2 is even, so set up t to start on d samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = reflect(jnp.arange(-m2, r + m2, dtype=int), -0.5, r - 0.5)

        t = jnp.arange(3, r + m, 2)
        if jnp.sum(ha * hb) > 0:
            ta = t
            tb = t - 1
        else:
            ta = t - 1
            tb = t

        # Select odd and even samples from ha and hb. Note that due to 0-indexing
        # 'odd' and 'even' are not perhaps what you might expect them to be.
        hao = as_column_vector(ha[0:m:2])
        hae = as_column_vector(ha[1:m:2])
        hbo = as_column_vector(hb[0:m:2])
        hbe = as_column_vector(hb[1:m:2])

        s = jnp.arange(0, r * 2, 4)

        Y=Y.at[s, :].set(_column_convolve(X[xe[tb - 2], :], hae))
        Y=Y.at[s + 1, :].set(_column_convolve(X[xe[ta - 2], :], hbe))
        Y=Y.at[s + 2, :].set(_column_convolve(X[xe[tb], :], hao))
        Y=Y.at[s + 3, :].set(_column_convolve(X[xe[ta], :], hbo))
    else:
        # m/2 is odd, so set up t to start on b samples.
        # Set up vector for symmetric extension of X with repeated end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = reflect(jnp.arange(-m2, r + m2, dtype=int), -0.5, r - 0.5)

        t = jnp.arange(2, r + m - 1, 2)
        if jnp.sum(ha * hb) > 0:
            ta = t
            tb = t - 1
        else:
            ta = t - 1
            tb = t

        # Select odd and even samples from ha and hb. Note that due to 0-indexing
        # 'odd' and 'even' are not perhaps what you might expect them to be.
        hao = as_column_vector(ha[0:m:2])
        hae = as_column_vector(ha[1:m:2])
        hbo = as_column_vector(hb[0:m:2])
        hbe = as_column_vector(hb[1:m:2])

        s = jnp.arange(0, r * 2, 4)

        Y=Y.at[s, :].set(_column_convolve(X[xe[tb], :], hao))
        Y=Y.at[s + 1, :].set(_column_convolve(X[xe[ta], :], hbo))
        Y=Y.at[s + 2, :].set(_column_convolve(X[xe[tb], :], hae))
        Y=Y.at[s + 3, :].set(_column_convolve(X[xe[ta], :], hbe))

    return Y


# vim:sw=4:sts=4:et
