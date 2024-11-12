"""Unit tests for Buffa-Christansen spaces."""

import bempp_cl.api
import pytest
import numpy as np

local_vertices = [[0, 0], [1, 0], [0, 1]]


@pytest.mark.parametrize("space_type,degree", [("P", 1)])
def test_H1_continuity(space_type, degree):
    """Test that BC spaces have continuous normal components."""
    grid = bempp_cl.api.shapes.sphere(h=0.2)
    space = bempp_cl.api.function_space(grid, space_type, degree)
    f = bempp_cl.api.GridFunction(space, coefficients=range(space.global_dof_count))

    for e, (vertices, neighbors) in enumerate(zip(grid.edges.T, grid.edge_neighbors)):
        v0 = [local_vertices[list(grid.elements[:, neighbors[0]]).index(v)] for v in vertices]
        v1 = [local_vertices[list(grid.elements[:, neighbors[1]]).index(v)] for v in vertices]
        points0 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v0)])
        points1 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v1)])

        values0 = f.evaluate(neighbors[0], points0)
        values1 = f.evaluate(neighbors[1], points1)

        assert np.allclose(values0, values1)


@pytest.mark.parametrize("space_type", ["RT", "BC"])
def test_Hdiv_continuity(space_type):
    """Test that BC spaces have continuous normal components."""
    grid = bempp_cl.api.shapes.sphere(h=0.2)
    space = bempp_cl.api.function_space(grid, space_type, 0)
    f = bempp_cl.api.GridFunction(space, coefficients=range(space.global_dof_count))

    if space_type == "BC":
        grid = grid.barycentric_refinement

    for e, (vertices, neighbors) in enumerate(zip(grid.edges.T, grid.edge_neighbors)):
        v0 = [local_vertices[list(grid.elements[:, neighbors[0]]).index(v)] for v in vertices]
        v1 = [local_vertices[list(grid.elements[:, neighbors[1]]).index(v)] for v in vertices]
        points0 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v0)])
        points1 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v1)])

        tangent = grid.vertices[:, vertices[1]] - grid.vertices[:, vertices[0]]
        tangent /= np.linalg.norm(tangent)
        normal0 = np.cross(tangent, grid.normals[neighbors[0]])
        normal1 = np.cross(tangent, grid.normals[neighbors[1]])

        assert np.isclose(np.linalg.norm(normal0), 1)
        assert np.isclose(np.linalg.norm(normal1), 1)

        values0 = f.evaluate(neighbors[0], points0)
        values1 = f.evaluate(neighbors[1], points1)

        components0 = [normal0.dot(v) for v in values0.T]
        components1 = [normal1.dot(v) for v in values1.T]

        assert np.allclose(components0, components1)


@pytest.mark.parametrize("space_type", ["NC", "RBC"])
def test_Hcurl_continuity(space_type):
    """Test that BC spaces have continuous normal components."""
    grid = bempp_cl.api.shapes.sphere(h=0.2)
    space = bempp_cl.api.function_space(grid, space_type, 0)
    f = bempp_cl.api.GridFunction(space, coefficients=range(space.global_dof_count))

    if space_type == "RBC":
        grid = grid.barycentric_refinement

    for e, (vertices, neighbors) in enumerate(zip(grid.edges.T, grid.edge_neighbors)):
        v0 = [local_vertices[list(grid.elements[:, neighbors[0]]).index(v)] for v in vertices]
        v1 = [local_vertices[list(grid.elements[:, neighbors[1]]).index(v)] for v in vertices]
        points0 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v0)])
        points1 = np.array([[a + i / 5 * (b - a) for i in range(6)] for a, b in zip(*v1)])

        tangent = grid.vertices[:, vertices[1]] - grid.vertices[:, vertices[0]]
        tangent /= np.linalg.norm(tangent)

        values0 = f.evaluate(neighbors[0], points0)
        values1 = f.evaluate(neighbors[1], points1)

        components0 = [tangent.dot(v) for v in values0.T]
        components1 = [tangent.dot(v) for v in values1.T]

        assert np.allclose(components0, components1)
