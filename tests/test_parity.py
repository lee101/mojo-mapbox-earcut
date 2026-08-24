import numpy as np
import pytest

import mapbox_earcut as upstream
import mojo_mapbox_earcut as mojo


def triangle_area(vertices, triangles):
    total = 0.0
    for a, b, c in triangles.reshape(-1, 3):
        pa, pb, pc = vertices[[a, b, c]]
        total += abs((pb[0] - pa[0]) * (pc[1] - pa[1]) -
                     (pb[1] - pa[1]) * (pc[0] - pa[0])) * 0.5
    return total


def assert_parity(vertices, ends, fn):
    got = fn(vertices, ends)
    expected = getattr(upstream, fn.__name__)(vertices, ends)
    assert got.dtype == expected.dtype == np.uint32
    assert got.size == expected.size
    assert got.size % 3 == 0
    assert np.all(got < len(vertices))
    assert triangle_area(vertices.astype(np.float64), got) == pytest.approx(
        triangle_area(vertices.astype(np.float64), expected), rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("fn,dtype", [
    (mojo.triangulate_float32, np.float32),
    (mojo.triangulate_float64, np.float64),
    (mojo.triangulate_int32, np.int32),
    (mojo.triangulate_int64, np.int64),
])
def test_typed_entry_points_match_upstream_on_convex_ring(fn, dtype):
    vertices = np.array([[0, 0], [8, 0], [8, 5], [0, 5]], dtype=dtype)
    assert_parity(vertices, np.array([4], dtype=np.uint32), fn)


def test_concave_ring_matches_upstream_geometry():
    vertices = np.array([[0, 0], [7, 0], [7, 7], [4, 3], [2, 6], [0, 7]], dtype=np.float64)
    assert_parity(vertices, np.array([6], dtype=np.uint32), mojo.triangulate_float64)


def test_clockwise_ring_matches_upstream_geometry():
    vertices = np.array([[0, 0], [0, 5], [8, 5], [8, 0]], dtype=np.float64)
    assert_parity(vertices, np.array([4], dtype=np.uint32), mojo.triangulate_float64)


def test_simd_linked_list_tail_matches_upstream_geometry():
    vertices = np.array([[0, 0], [8, 0], [10, 4], [5, 8], [0, 5]], dtype=np.float64)
    assert_parity(vertices, np.array([5], dtype=np.uint32), mojo.triangulate_float64)


@pytest.mark.parametrize("fn,dtype", [
    (mojo.triangulate_float32, np.float32),
    (mojo.triangulate_float64, np.float64),
    (mojo.triangulate_int32, np.int32),
    (mojo.triangulate_int64, np.int64),
])
def test_rectangle_hole_matches_upstream_geometry(fn, dtype):
    vertices = np.array([
        [0, 0], [10, 0], [10, 10], [0, 10],
        [2, 2], [2, 8], [8, 8], [8, 2],
    ], dtype=dtype)
    assert_parity(vertices, np.array([4, 8], dtype=np.uint32), fn)


def test_two_holes_match_upstream_geometry():
    vertices = np.array([
        [0, 0], [20, 0], [20, 20], [0, 20],
        [2, 2], [2, 6], [6, 6], [6, 2],
        [12, 12], [12, 17], [17, 17], [17, 12],
    ], dtype=np.float64)
    assert_parity(vertices, np.array([4, 8, 12], dtype=np.uint32), mojo.triangulate_float64)


def test_collinear_and_repeated_vertices_match_upstream_area():
    vertices = np.array([[0, 0], [4, 0], [4, 0], [4, 4], [2, 4], [2, 4], [0, 4]], dtype=np.float64)
    assert_parity(vertices, np.array([7], dtype=np.uint32), mojo.triangulate_float64)


@pytest.mark.parametrize("n", [80, 81, 800, 801])
def test_large_concave_star_matches_upstream_geometry(n):
    angle = np.arange(n) * (2 * np.pi / n)
    radius = np.where(np.arange(n) % 2, 4.0, 10.0)
    vertices = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    assert_parity(vertices, np.array([n], dtype=np.uint32), mojo.triangulate_float64)


def test_spatial_index_with_hole_matches_upstream_geometry():
    n = 96
    angle = (np.arange(n) + 0.13) * (2 * np.pi / n)
    radius = 50 + 3 * np.sin(5 * angle)
    outer = np.column_stack((radius * np.cos(angle), 0.83 * radius * np.sin(angle)))
    hole = np.array([[-7.3, -4.1], [-6.2, 6.4], [8.7, 5.2], [7.1, -5.8]])
    vertices = np.vstack((outer, hole))
    assert_parity(vertices, np.array([n, n + 4], dtype=np.uint32), mojo.triangulate_float64)


def test_empty_input_matches_upstream():
    vertices = np.empty((0, 2), dtype=np.float64)
    ends = np.empty(0, dtype=np.uint32)
    assert np.array_equal(mojo.triangulate_float64(vertices, ends), upstream.triangulate_float64(vertices, ends))


@pytest.mark.parametrize("ends", [np.array([3, 2]), np.array([2]), np.array([0])])
def test_bad_ring_ends_raise_value_error(ends):
    vertices = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
    with pytest.raises(ValueError):
        mojo.triangulate_float64(vertices, ends)


def test_bad_vertex_shape_raises_value_error():
    with pytest.raises(ValueError, match="shape"):
        mojo.triangulate_float64(np.array([0.0, 1.0]), np.array([1]))


def test_non_contiguous_inputs_are_copied_safely():
    vertices = np.array([[0, 99, 0, 99], [8, 99, 0, 99], [8, 99, 5, 99], [0, 99, 5, 99]], dtype=np.float64)[:, ::2]
    ends = np.array([4, 99], dtype=np.uint32)[::2]
    assert_parity(vertices, ends, mojo.triangulate_float64)


def test_lossy_coordinate_or_ring_index_conversion_is_rejected():
    vertices = np.array([[2**53 + 1, 0], [2**53 + 3, 0], [2**53 + 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="float64 kernel"):
        mojo.triangulate_int64(vertices, np.array([3], dtype=np.uint32))
    with pytest.raises(ValueError, match="fit in uint32"):
        mojo.triangulate_float64(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64), np.array([2**32 + 3], dtype=np.int64))


def test_c_abi_rejects_null_pointers_before_dereferencing():
    assert mojo._lib.lib().mme_triangulate_f64(*([0] * 10)) == -1
