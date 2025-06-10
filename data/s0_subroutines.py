import numpy as np

"""
s0_subroutines.py
Utility subroutines used across EWS processing modules,
primarily for run-length encoding of binary or numeric sequences.
"""

def rle(sequence, series=None):
    """
    Perform run-length encoding on a 1D sequence.

    This function scans `sequence` for consecutive runs of identical
    values, and returns an array of (value, length) pairs. If `series`
    is provided, only lengths corresponding to runs of that specific
    value are returned.

    Parameters:
    -----------
    sequence : iterable
        Input sequence of hashable values (e.g., ints, strings).
    series : optional
        If provided, return only the lengths for runs equal to `series`.

    Returns:
    --------
    np.ndarray
        If series is None:
            Array of shape (num_runs, 2), where each row is [value, run_length].
        If series is given:
            1D array of lengths for runs matching the specified series value.
    """
    # Use itertools.groupby to identify consecutive runs of identical items
    from itertools import groupby
    # Build an array of (value, run_length)
    a_rle = np.asarray([(key, sum(1 for _ in group))
                        for key, group in groupby(sequence)])

    if series is not None:
        # Filter to only runs where the value equals `series`, then return lengths
        return a_rle[a_rle[:,0] == series][:,1]
    else:
        # Return full run-length encoding array
        return a_rle
