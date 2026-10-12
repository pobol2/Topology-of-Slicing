"""Hilbert traversals of dyadic grids (Skilling's transform)."""

from __future__ import annotations

import numpy as np


def _distance_to_transpose(
    distance: int,
    order: int,
    dimensions: int,
) -> list[int]:
    """Convert a Hilbert distance to its transposed bit representation."""
    bit_string = f"{distance:0{order * dimensions}b}"
    return [int(bit_string[index::dimensions], 2) for index in range(dimensions)]


def _point_from_distance(
    distance: int,
    order: int,
    dimensions: int,
) -> list[int]:
    """Convert one Hilbert index to integer grid coordinates.

    This implements John Skilling's integer Hilbert transform.
    """
    coordinates = _distance_to_transpose(
        distance,
        order,
        dimensions,
    )

    # Gray decode.
    temporary = coordinates[dimensions - 1] >> 1
    for index in range(dimensions - 1, 0, -1):
        coordinates[index] ^= coordinates[index - 1]
    coordinates[0] ^= temporary

    q = 2
    limit = 2 << (order - 1)
    while q != limit:
        p = q - 1
        for index in range(dimensions - 1, -1, -1):
            if coordinates[index] & q:
                coordinates[0] ^= p
            else:
                temporary = (coordinates[0] ^ coordinates[index]) & p
                coordinates[0] ^= temporary
                coordinates[index] ^= temporary
        q <<= 1

    return coordinates


def hilbert_nd(order: int, dimensions: int) -> np.ndarray:
    """Skilling traversal of unit-hypercube cell centres."""
    if order < 1:
        raise ValueError("order must be at least 1.")
    if dimensions < 2:
        raise ValueError("dimensions must be at least 2.")
    cells_per_axis = 2**order
    integer_points = np.asarray(
        [
            _point_from_distance(distance, order, dimensions)
            for distance in range(cells_per_axis**dimensions)
        ],
        dtype=float,
    )
    return (integer_points + 0.5) / cells_per_axis


def hilbert_2d(order: int) -> np.ndarray:
    """Visit every centre in a 2D dyadic grid, in Hilbert order."""
    return hilbert_nd(order, 2)


def hilbert_3d(order: int) -> np.ndarray:
    """Generate an order-``order`` 3D Hilbert traversal in the unit cube.

    The curve visits the centers of every cell in a
    ``2**order x 2**order x 2**order`` grid. The returned array has shape
    ``(8**order, 3)`` and its row order is the Hilbert path order.
    """
    return hilbert_nd(order, 3)
