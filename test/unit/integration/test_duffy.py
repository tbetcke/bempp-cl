"""Test routines related to duffy rules."""

from bempp_cl.api.integration import duffy_galerkin as _duffy

_order = 6


def test_number_of_quad_points_for_coincident_case():
    """Test for correct number of points in coincident case."""

    actual_number_of_points = _duffy.number_of_quadrature_points(_order, "coincident")
    expected_number_of_points = 6 * _order**4

    assert actual_number_of_points == expected_number_of_points


def test_number_of_quad_points_for_edge_adjacent_case():
    """Test for correct number of points in edge adjacent case."""

    actual_number_of_points = _duffy.number_of_quadrature_points(_order, "edge_adjacent")
    expected_number_of_points = 5 * _order**4

    assert actual_number_of_points == expected_number_of_points


def test_number_of_quad_points_for_vertex_adjacent_case():
    """Test for correct number of points in vertex adjacent case."""

    actual_number_of_points = _duffy.number_of_quadrature_points(_order, "vertex_adjacent")
    expected_number_of_points = 2 * _order**4

    assert actual_number_of_points == expected_number_of_points
