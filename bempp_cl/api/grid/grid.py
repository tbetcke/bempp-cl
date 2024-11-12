"""The basic grid class."""

from bempp_cl.helpers import timeit as _timeit
import collections as _collections

import numba as _numba
import numpy as _np

EDGES_ID = 2
VERTICES_ID = 1

_EDGE_LOCAL = _np.array([[0, 1], [2, 0], [1, 2]])


class Grid(object):
    """The Grid class."""

    @_timeit
    def __init__(self, vertices, elements, domain_indices=None, grid_id=None, scatter=True):
        """Create a grid from a vertices and an elements array."""
        from bempp_cl.api import log
        from bempp_cl.api.utils import pool
        from bempp_cl.api.utils.helpers import create_unique_id

        self._vertices = None
        self._elements = None
        self._domain_indices = None
        self._edges = None
        self._element_edges = None
        self._edge_adjacency = None
        self._vertex_adjacency = None
        self._element_neighbors = None
        self._vertex_on_boundary = None
        self._edge_on_boundary = None
        self._edge_neighbors = None
        self._vertex_neighbors = None
        self._barycentric_grid = None
        if grid_id:
            self._id = grid_id
        else:
            self._id = create_unique_id()

        self._volumes = None
        self._normals = None
        self._jacobians = None
        self._jacobian_inverse_transposed = None
        self._diameters = None
        self._integration_elements = None
        self._centroids = None

        self._device_interfaces = {}

        self._element_to_vertex_matrix = None
        self._element_to_element_matrix = None

        self._normalize_and_assign_input(vertices, elements, domain_indices)
        self._enumerate_edges()

        self._get_element_adjacency_for_edges_and_vertices()
        self._compute_geometric_quantities()

        self._compute_boundary_information()

        self._compute_edge_neighbors()
        self._compute_vertex_neighbors()

        self._grid_data_double = GridDataDouble(
            self._vertices,
            self._elements,
            self._edges,
            self._element_edges,
            self._volumes,
            self._normals,
            self._jacobians,
            self._jacobian_inverse_transposed,
            self._diameters,
            self._integration_elements,
            self._centroids,
            self._domain_indices,
            self._vertex_on_boundary,
            self._element_neighbors.indices,
            self._element_neighbors.indexptr,
        )

        self._grid_data_single = GridDataFloat(
            self._vertices.astype("float32"),
            self._elements,
            self._edges,
            self._element_edges,
            self._volumes.astype("float32"),
            self._normals.astype("float32"),
            self._jacobians.astype("float32"),
            self._jacobian_inverse_transposed.astype("float32"),
            self._diameters.astype("float32"),
            self._integration_elements.astype("float32"),
            self._centroids.astype("float32"),
            self._domain_indices,
            self._vertex_on_boundary,
            self._element_neighbors.indices,
            self._element_neighbors.indexptr,
        )

        self._is_scattered = False

        if scatter and pool.is_initialised() and not pool.is_worker():
            self._scatter()
        if not pool.is_worker():
            log(
                (
                    f"Created grid with id {self.id}. Elements: {self.number_of_elements}. "
                    + f"Edges: {self.number_of_edges}. Vertices: {self.number_of_vertices}"
                )
            )

    @property
    def vertex_adjacency(self):
        """
        Vertex adjacency information.

        Returns a matrix with 4 rows. Each column has the entries e0,
        e1, ind0, ind1, which means that element e0 is connected to
        element e1 via local vertex index ind0 in e0 and ind1 in e1.
        Only returnes connectivity via a single vertex. For
        connectivity via edges see edge_adjacency.

        """
        return self._vertex_adjacency

    @property
    def edge_adjacency(self):
        """
        Edge adjacency information.

        Returns a matrix with 6 rows. Each column has the entries e0,
        e1, v00, v01, v11, v12, which means that element e0 is
        connected to element e1. Vertex v00 in element e0 is
        identical to vertex v11 in element e1, and vertex v01 in
        element 0 is identical to vertex v12 in element e1.
        """
        return self._edge_adjacency

    @property
    def element_to_vertex_matrix(self):
        """Return the matrix mapping vertices to elements."""
        return self._element_to_vertex_matrix

    @property
    def element_to_element_matrix(self):
        """
        Return element to element matrix.

        If entry (i,j) has the value n > 0, element i
        and element j are connected via n vertices.
        """
        return self._element_to_element_matrix

    @property
    def element_neighbors(self):
        """
        Return named tuple (indices, indexptr).

        The neighbors of element i are given as
        element_neighbors.indices[
            element_neighbors.indptr[i] : element_neighbors.indptr[i +1]].
        Note that the element i is contained in the list of neighbors.

        """
        return self._element_neighbors

    @property
    def number_of_vertices(self):
        """Return number of vertices."""
        return self._vertices.shape[1]

    @property
    def number_of_edges(self):
        """Return number of edges."""
        return self._edges.shape[1]

    @property
    def number_of_elements(self):
        """Return number of elements."""
        return self._elements.shape[1]

    @property
    def vertices(self):
        """Return vertices."""
        return self._vertices

    @property
    def elements(self):
        """Return elements."""
        return self._elements

    @property
    def edges(self):
        """Return edges."""
        return self._edges

    @property
    def centroids(self):
        """Return the centroids of the elements."""
        return self._centroids

    @property
    def domain_indices(self):
        """Return domain indices."""
        return self._domain_indices

    @property
    def element_edges(self):
        """
        Return an array of edge indices for each element.

        element_edges[i, j] is the index of the ith edge
        in the jth element.

        """
        return self._element_edges

    @property
    def device_interfaces(self):
        """Return the dictionary of device interfaces for the grid."""
        return self._device_interfaces

    @property
    def as_array(self):
        """
        Convert the grid to an array.

        For a grid with N elements returns a 1d array with
        9 * N entries. The three nodes for element with index e
        can be found in [9 * e, 9 * (e + 1)].

        """
        return self.vertices.T[self.elements.flatten(order="F"), :].flatten(order="C")

    @property
    def bounding_box(self):
        """
        Return the bounding box for the grid.

        The bounding box is a 3x2 array box such that
        box[:, 0] contains (xmin, ymin, zmin) and box[:, 1]
        contains (xmax, ymax, zmax).

        """
        box = _np.empty((3, 2), dtype="float64")
        box[:, 0] = _np.min(self.vertices, axis=1)
        box[:, 1] = _np.max(self.vertices, axis=1)

        return box

    @property
    def volumes(self):
        """Return element volumes."""
        return self._volumes

    @property
    def diameters(self):
        """Return element diameters."""
        return self._diameters

    @property
    def maximum_element_diameter(self):
        """Return the maximum element diameter."""
        return _np.max(self.diameters)

    @property
    def minimum_element_diameter(self):
        """Return the maximum element diameter."""
        return _np.min(self.diameters)

    @property
    def normals(self):
        """Return normals."""
        return self._normals

    @property
    def jacobians(self):
        """Return Jacobians."""
        return self._jacobians

    @property
    def integration_elements(self):
        """Return integration elements."""
        return self._integration_elements

    @property
    def jacobian_inverse_transposed(self):
        """Return the jacobian inverse transposed."""
        return self._jacobian_inverse_transposed

    @property
    def vertex_on_boundary(self):
        """Return vertex boundary information."""
        return self._vertex_on_boundary

    @property
    def edge_on_boundary(self):
        """Return edge boundary information."""
        return self._edge_on_boundary

    @property
    def edge_neighbors(self):
        """Return for each edge the list of neighboring elements.."""
        return self._edge_neighbors

    def data(self, precision="double"):
        """Return Numba container with all relevant grid data."""
        if precision == "double":
            return self._grid_data_double
        elif precision == "single":
            return self._grid_data_single
        else:
            raise ValueError("precision must be one of 'single', 'double'")

    @property
    def vertex_neighbors(self):
        """Return for each vertex the list of neighboring elements."""
        return self._vertex_neighbors

    @property
    def barycentric_refinement(self):
        """Return the barycentric refinement of this grid."""
        if self._barycentric_grid is None:
            self._barycentric_grid = barycentric_refinement(self)
        return self._barycentric_grid

    @property
    def id(self):
        """Return a unique id for the grid."""
        return self._id

    def _scatter(self):
        """Initialise the grid on all workers."""
        from bempp_cl.api.utils import pool

        array_proxies = pool.to_buffer(self.vertices, self.elements, self.domain_indices)

        pool.execute(_grid_scatter_worker, self.id, array_proxies)
        self._is_scattered = True

    def entity_count(self, codim):
        """Return the number of entities of given codimension."""
        if codim == 0:
            return self.number_of_elements
        if codim == 1:
            return self.number_of_edges
        if codim == 2:
            return self.number_of_vertices

        raise ValueError("codim must be one of 0, 1, or 2.")

    def plot(self):
        """Plot the grid."""
        from bempp_cl.api.external.viewers import visualize

        visualize(self)

    def get_element(self, index):
        """Return element with a given index."""
        return Element(self, index)

    def entity_iterator(self, codim):
        """Return an iterator for a given codim."""

        def element_iterator():
            """Iterate over elements."""
            for index in range(self.number_of_elements):
                yield Element(self, index)

        def vertex_iterator():
            """Iterate over vertices."""
            for index in range(self.number_of_vertices):
                yield Vertex(self, index)

        def edge_iterator():
            """Iterate over edges."""
            for index in range(self.number_of_edges):
                yield Edge(self, index)

        if codim not in [0, 1, 2]:
            raise ValueError("codim must be one of 0, 1, or 2.")

        if codim == 0:
            iterator = element_iterator()
        elif codim == 1:
            iterator = edge_iterator()
        elif codim == 2:
            iterator = vertex_iterator()

        return iterator

    def map_to_point_cloud(self, order=None, local_points=None, precision="double"):
        """
        Return a point cloud representation of the grid on quadratur points.

        Return a representation of the grid as a point cloud using points on
        each element either defined through a triangle Gauss qudrature order
        or by directly specifying an array of local points.

        Parameters
        ----------
        order : Integer
            Optional parameter. Specify a quadrature order for the point
            cloud generation.
        local_points: Numpy array
            A 2 x N array of N points in local reference coordinates that specify
            the points to use for each triangle.
        precision: String
            Either 'single' or 'double'.

        If neither order nor local_points is specified the quadrature order is
        obtained from the global parameters.

        Returns a M x 3 array of M points that represent the grid on the specified
        points.

        """
        import bempp_cl.api
        from bempp_cl.api.integration.triangle_gauss import rule

        if local_points is None:
            if order is None:
                order = bempp_cl.api.GLOBAL_PARAMETERS.quadrature.regular
            local_points, _ = rule(order)

        return grid_to_points(self.data("double"), local_points)

    def refine(self):
        """Return a new grid with all elements refined."""
        new_number_of_vertices = self.number_of_edges + self.number_of_vertices

        new_vertices = _np.empty((3, new_number_of_vertices), dtype="float64", order="F")

        new_vertices[:, : self.number_of_vertices] = self.vertices

        # Each edge midpoint forms a new vertex.
        new_vertices[:, self.number_of_vertices :] = 0.5 * (
            self.vertices[:, self.edges[0, :]] + self.vertices[:, self.edges[1, :]]
        )

        new_elements = _np.empty((3, 4 * self.number_of_elements), order="F", dtype="uint32")

        new_domain_indices = _np.repeat(self.domain_indices, 4)

        for index, elem in enumerate(self.elements.T):
            vertex0 = elem[0]
            vertex1 = elem[1]
            vertex2 = elem[2]
            vertex01 = self.element_edges[0, index] + self.number_of_vertices
            vertex20 = self.element_edges[1, index] + self.number_of_vertices
            vertex12 = self.element_edges[2, index] + self.number_of_vertices

            new_elements[:, 4 * index] = [vertex0, vertex01, vertex20]

            new_elements[:, 4 * index + 1] = [vertex01, vertex1, vertex12]

            new_elements[:, 4 * index + 2] = [vertex12, vertex2, vertex20]

            new_elements[:, 4 * index + 3] = [vertex01, vertex12, vertex20]

        return Grid(new_vertices, new_elements, new_domain_indices)

    def _compute_vertex_neighbors(self):
        """Return all elements adjacent to a given vertex."""
        from bempp_cl.helpers import IndexList

        # self._vertex_neighbors = [None for _ in range(self.number_of_vertices)]

        indptr = self.element_to_vertex_matrix.indptr
        indices = self.element_to_vertex_matrix.indices
        self._vertex_neighbors = IndexList(indices, indptr)
        # for index in range(self.number_of_vertices):
        #    self._vertex_neighbors[index] = indices[indptr[index] : indptr[index + 1]]

    def _normalize_and_assign_input(self, vertices, elements, domain_indices):
        """Convert input into the right form."""
        from bempp_cl.api.utils.helpers import align_array

        if domain_indices is None:
            domain_indices = _np.zeros(elements.shape[1], dtype="uint32")

        self._vertices = align_array(vertices, "float64", "F")
        self._elements = align_array(elements, "uint32", "F")
        self._domain_indices = align_array(domain_indices, "uint32", "F")

    def _enumerate_edges(self):
        """
        Enumerate all edges in a given grid.

        Assigns a tuple (edges, element_edges) to
        self._edges and self._element_edges.
        element_edges is an array a such that a[i, j] is the
        index of the ith edge in the jth elements, and edges
        is a 2 x nedges array such that the jth column stores the
        two nodes associated with the jth edge.

        """
        # The following would be better defined inside the njitted routiine.
        # But Numba then throws an error that it cannot find the UniTuple type.
        edge_tuple_to_index = _numba.typed.Dict.empty(
            key_type=_numba.types.containers.UniTuple(_numba.types.int64, 2),
            value_type=_numba.types.int64,
        )

        self._edges, self._element_edges = _numba_enumerate_edges(self._elements, edge_tuple_to_index)

    def _get_element_adjacency_for_edges_and_vertices(self):
        """Get element adjacency.

        The array edge_adjacency has 6 rows, such that for index j the
        element edge_adjacency[0, j] is connected with element
        edge_adjacency[1, j] via the vertices edge_adjacency[2:4, j]
        in the first element and the vertices edge_adjacency[4:6, j]
        in the second element. The vertex numbers here are local
        numbers (0, 1 or 2).

        The array vertex_adjacency has 4 rows, such that for index j the
        element vertex_adjacency[0, j] is connected with
        vertex_adjacency[1, j] via the vertex vertex_adjacency[2, j]
        in the first element and the vertex vertex_adjacency[3, j]
        in the second element. The vertex numbers here are local numbers
        (0, 1 or 2).

        """
        from bempp_cl.helpers import IndexList

        self._element_to_vertex_matrix = get_element_to_vertex_matrix(self._vertices, self._elements)

        elem_to_elem_matrix = get_element_to_element_matrix(self._vertices, self._elements)

        self._element_to_element_matrix = elem_to_elem_matrix

        elements1, elements2, nvertices = _get_element_to_element_vertex_count(elem_to_elem_matrix)

        vertex_connected_elements1, vertex_connected_elements2 = _element_filter(
            elements1, elements2, nvertices, VERTICES_ID
        )

        edge_connected_elements1, edge_connected_elements2 = _element_filter(elements1, elements2, nvertices, EDGES_ID)

        self._vertex_adjacency = _find_vertex_adjacency(
            self._elements, vertex_connected_elements1, vertex_connected_elements2
        )

        self._edge_adjacency = _find_edge_adjacency(self._elements, edge_connected_elements1, edge_connected_elements2)

        self._element_neighbors = IndexList(elem_to_elem_matrix.indices, elem_to_elem_matrix.indptr)

    def _compute_geometric_quantities(self):
        """Compute geometric quantities for the grid."""
        element_vertices = self.vertices.T[self.elements.flatten(order="F")]
        indexptr = 3 * _np.arange(self.number_of_elements)
        indices = _np.repeat(indexptr, 2) + _np.tile([1, 2], self.number_of_elements)

        centroids = 1.0 / 3 * _np.sum(_np.reshape(element_vertices, (self.number_of_elements, 3, 3)), axis=1)

        jacobians = (element_vertices - _np.repeat(element_vertices[::3], 3, axis=0))[indices]

        normal_directions = _np.cross(jacobians[::2], jacobians[1::2], axis=1)
        normal_direction_norms = _np.linalg.norm(normal_directions, axis=1)
        normals = normal_directions / _np.expand_dims(normal_direction_norms, 1)

        volumes = 0.5 * normal_direction_norms

        jacobian_diff = jacobians[::2] - jacobians[1::2]
        diff_norms = _np.linalg.norm(jacobian_diff, axis=1)
        jac_vector_norms = _np.linalg.norm(jacobians, axis=1)

        diameters = jac_vector_norms[::2] * jac_vector_norms[1::2] * diff_norms / normal_direction_norms

        self._volumes = volumes
        self._normals = normals
        self._jacobians = _np.swapaxes(_np.reshape(jacobians, (self.number_of_elements, 2, 3)), 1, 2)
        self._diameters = diameters
        self._centroids = centroids

        jac_transpose_jac = _np.empty((self.number_of_elements, 2, 2), dtype="float64")
        for index in range(self.number_of_elements):
            jac_transpose_jac[index] = self.jacobians[index].T.dot(self.jacobians[index])
        self._integration_elements = _np.sqrt(_np.linalg.det(jac_transpose_jac))

        jac_transpose_jac_inv = _np.linalg.inv(jac_transpose_jac)

        self._jacobian_inverse_transposed = _np.empty((self.number_of_elements, 3, 2), dtype="float64")

        for index in range(self.number_of_elements):
            self._jacobian_inverse_transposed[index] = self.jacobians[index].dot(jac_transpose_jac_inv[index])

    def _compute_boundary_information(self):
        """
        Return a boolean array with boundary information.

        Computes arr0, arr1 such that arr0[j] is True if
        vertex j lies on the boundary and arr1[i] is True if edge
        i lies on the boundary.
        """
        from scipy.sparse import csr_matrix

        element_edges = self.element_edges

        number_of_elements = self.number_of_elements
        number_of_edges = self.number_of_edges
        number_of_vertices = self.number_of_vertices
        edge_indices = _np.ravel(element_edges, order="F")
        repeated_element_indices = _np.repeat(_np.arange(number_of_elements), 3)
        data = _np.ones(3 * number_of_elements, dtype="uint32")

        element_to_edge = csr_matrix(
            (data, (repeated_element_indices, edge_indices)),
            shape=(number_of_elements, number_of_edges),
        )

        edge_to_edge = element_to_edge.T.dot(element_to_edge)
        arr1 = edge_to_edge.diagonal() == 1
        arr0 = _np.full(number_of_vertices, False)

        for boundary_edge_index in _np.flatnonzero(arr1):
            arr0[self.edges[:, boundary_edge_index]] = True

        self._vertex_on_boundary = arr0
        self._edge_on_boundary = arr1

    def _compute_edge_neighbors(self):
        """Get the neighbors of each edge."""
        edge_neighbors = [[] for _ in range(self.number_of_edges)]

        for element_index in range(self.number_of_elements):
            for local_index in range(3):
                edge_neighbors[self.element_edges[local_index, element_index]].append(element_index)
        self._edge_neighbors = [tuple(elem) for elem in edge_neighbors]


@_numba.experimental.jitclass(
    [
        ("vertices", _numba.float64[:, :]),
        ("elements", _numba.uint32[:, :]),
        ("edges", _numba.uint32[:, :]),
        ("element_edges", _numba.uint32[:, :]),
        ("volumes", _numba.float64[:]),
        ("normals", _numba.float64[:, :]),
        ("jacobians", _numba.float64[:, :, :]),
        ("jac_inv_trans", _numba.float64[:, :, :]),
        ("diameters", _numba.float64[:]),
        ("integration_elements", _numba.float64[:]),
        ("centroids", _numba.float64[:, :]),
        ("domain_indices", _numba.uint32[:]),
        ("vertex_on_boundary", _numba.boolean[:]),
        ("element_neighbor_indices", _numba.uint32[:]),
        ("element_neighbor_indexptr", _numba.uint32[:]),
    ]
)
class GridDataDouble(object):
    """A Numba container class for the grid data."""

    def __init__(
        self,
        vertices,
        elements,
        edges,
        element_edges,
        volumes,
        normals,
        jacobians,
        jac_inv_trans,
        diameters,
        integration_elements,
        centroids,
        domain_indices,
        vertex_on_boundary,
        element_neighbor_indices,
        element_neighbor_indexptr,
    ):
        """Create a GridDataDouble."""
        self.vertices = vertices
        self.elements = elements
        self.edges = edges
        self.element_edges = element_edges
        self.volumes = volumes
        self.normals = normals
        self.jacobians = jacobians
        self.jac_inv_trans = jac_inv_trans
        self.diameters = diameters
        self.integration_elements = integration_elements
        self.centroids = centroids
        self.domain_indices = domain_indices
        self.vertex_on_boundary = vertex_on_boundary
        self.element_neighbor_indices = element_neighbor_indices
        self.element_neighbor_indexptr = element_neighbor_indexptr

    def local2global(self, elem_index, local_coords):
        """Map local to global coordinates."""
        return _np.expand_dims(self.vertices[:, self.elements[0, elem_index]], 1) + self.jacobians[elem_index].dot(
            local_coords
        )


@_numba.experimental.jitclass(
    [
        ("vertices", _numba.float32[:, :]),
        ("elements", _numba.uint32[:, :]),
        ("edges", _numba.uint32[:, :]),
        ("element_edges", _numba.uint32[:, :]),
        ("volumes", _numba.float32[:]),
        ("normals", _numba.float32[:, :]),
        ("jacobians", _numba.float32[:, :, :]),
        ("jac_inv_trans", _numba.float32[:, :, :]),
        ("diameters", _numba.float32[:]),
        ("integration_elements", _numba.float32[:]),
        ("centroids", _numba.float32[:, :]),
        ("domain_indices", _numba.uint32[:]),
        ("vertex_on_boundary", _numba.boolean[:]),
        ("element_neighbor_indices", _numba.uint32[:]),
        ("element_neighbor_indexptr", _numba.uint32[:]),
    ]
)
class GridDataFloat(object):
    """A Numba container class for the grid data."""

    def __init__(
        self,
        vertices,
        elements,
        edges,
        element_edges,
        volumes,
        normals,
        jacobians,
        jac_inv_trans,
        diameters,
        integration_elements,
        centroids,
        domain_indices,
        vertex_on_boundary,
        element_neighbor_indices,
        element_neighbor_indexptr,
    ):
        """Create a GridDataFloat."""
        self.vertices = vertices
        self.elements = elements
        self.edges = edges
        self.element_edges = element_edges
        self.volumes = volumes
        self.normals = normals
        self.jacobians = jacobians
        self.jac_inv_trans = jac_inv_trans
        self.diameters = diameters
        self.integration_elements = integration_elements
        self.centroids = centroids
        self.domain_indices = domain_indices
        self.vertex_on_boundary = vertex_on_boundary
        self.element_neighbor_indices = element_neighbor_indices
        self.element_neighbor_indexptr = element_neighbor_indexptr

    def local2global(self, elem_index, local_coords):
        """Map local to global coordinates."""
        return _np.expand_dims(self.vertices[:, self.elements[0, elem_index]], 1) + self.jacobians[elem_index].dot(
            local_coords
        )


class ElementGeometry(object):
    """Provides geometry information for an element."""

    def __init__(self, grid, index):
        """Initialize geometry wth a 3x3 array of corners."""
        self._grid = grid
        self._index = index

    @property
    def corners(self):
        """Return corners."""
        return self._grid.vertices[:, self._grid.elements[:, self._index]]

    @property
    def jacobian(self):
        """Return jacobian."""
        return self._grid.jacobians[self._index]

    @property
    def integration_element(self):
        """Return integration element."""
        return self._grid.integration_elements[self._index]

    @property
    def jacobian_inverse_transposed(self):
        """Return Jacobian inverse transposed."""
        return self._grid.jacobian_inverse_transposed[self._index]

    @property
    def normal(self):
        """Return normal."""
        return self._grid.normals[self._index]

    @property
    def volume(self):
        """Return volume."""
        return self._grid.volumes[self._index]

    @property
    def diameter(self):
        """Return the diameter of the circumcircle."""
        return self._grid.diameters[self._index]

    @property
    def centroid(self):
        """Return the centroid of the element."""
        return self._grid.centroids[self._index]

    def local2global(self, points):
        """Map points in local coordinates to global."""
        return _np.expand_dims(self.corners[:, 0], 1) + self.jacobian @ points


class Element(object):
    """Provides a view onto an element of the grid."""

    def __init__(self, grid, index):
        self._grid = grid
        self._index = index

    @property
    def index(self):
        """Index of the element."""
        return self._index

    @property
    def grid(self):
        """Associated grid."""
        return self._grid

    @property
    def geometry(self):
        """Return geometry."""
        grid = self._grid
        return ElementGeometry(grid, self.index)

    @property
    def domain_index(self):
        """Return the domain index."""
        return self._grid.domain_indices[self.index]

    def sub_entity_iterator(self, codim):
        """Return iterator over subentitites."""

        def edge_iterator():
            """Iterate over edges."""
            for index in self._grid.element_edges[:, self.index]:
                yield Edge(self._grid, index)

        def vertex_iterator():
            """Iterate over vertices."""
            for index in self._grid.elements[:, self.index]:
                yield Vertex(self._grid, index)

        if codim not in [1, 2]:
            raise ValueError("codim must be 1 (for edges) or 2 (for vertices)")

        if codim == 1:
            iterator = edge_iterator()

        if codim == 2:
            iterator = vertex_iterator()

        return iterator

    def __eq__(self, other):
        """Check if elements are equal."""
        if isinstance(other, Element):
            if other.grid == self.grid and other.index == self.index:
                return True
        return False


VertexGeometry = _collections.namedtuple("VertexGeometry", "corners")


class Vertex(object):
    """Provides a view onto a vertex of the grid."""

    def __init__(self, grid, index):
        """Create a vertex."""
        self._grid = grid
        self._index = index

    @property
    def index(self):
        """Index of the vertex."""
        return self._index

    @property
    def geometry(self):
        """Return geometry."""
        return VertexGeometry(self._grid.vertices[:, self.index].reshape(3, 1))


class EdgeGeometry(object):
    """Implementation of a geometry for edges."""

    def __init__(self, corners):
        """Create edge geometry."""
        self._corners = corners
        self._volume = _np.linalg.norm(corners[:, 1] - corners[:, 0])

    @property
    def corners(self):
        """Return the corners."""
        return self._corners

    @property
    def volume(self):
        """Return length of the edge."""
        return self._volume


class Edge(object):
    """Provides a view onto an edge of the grid."""

    def __init__(self, grid, index):
        """Create an edge."""
        self._grid = grid
        self._index = index

    @property
    def index(self):
        """Return the index of the edge."""
        return self._index

    @property
    def geometry(self):
        """Return geometry."""
        grid = self._grid
        return EdgeGeometry(grid.vertices[:, grid.edges[:, self.index]])


def get_element_to_vertex_matrix(vertices, elements):
    """Return the sparse matrix mapping vertices to elements."""
    from scipy.sparse import csr_matrix

    number_of_elements = elements.shape[1]
    number_of_vertices = vertices.shape[1]
    vertex_indices = _np.ravel(elements, order="F")
    vertex_element_indices = _np.repeat(_np.arange(number_of_elements), 3)
    data = _np.ones(len(vertex_indices), dtype="uint32")

    return csr_matrix(
        (data, (vertex_indices, vertex_element_indices)),
        shape=(number_of_vertices, number_of_elements),
        dtype="uint32",
    )


def get_element_to_element_matrix(vertices, elements):
    """
    Return element to element matrix.

    If entry (i,j) has the value n > 0, element i
    and element j are connected via n vertices.

    """
    element_to_vertex = get_element_to_vertex_matrix(vertices, elements)
    return element_to_vertex.T.dot(element_to_vertex)


@_numba.njit(locals={"index": _numba.types.int32})
def _compare_array_to_value(array, val):
    """
    Return i such that array[i] == val.

    If val not found return -1
    """
    for index, elem in enumerate(array):
        if elem == val:
            return index
    return -1


@_numba.njit(
    locals={
        "index1": _numba.types.int32,
        "index2": _numba.types.int32,
        "full_index1": _numba.types.int32,
    }
)
def _find_first_common_array_index_pair_from_position(array1, array2, start=0):
    """
    Return first index pair (i, j) such that array1[i] = array2[j].

    Assumes that one index pair satisfying the equality
    always exists. Method checks in array1 from position start
    onwards.
    """
    for index1 in range(len(array1[start:])):
        full_index1 = index1 + start
        index2 = _compare_array_to_value(array2, array1[full_index1])
        if index2 != -1:
            return (full_index1, index2)
    raise ValueError("Could not find a common index pair.")


@_numba.njit(locals={"offset": _numba.types.int32})
def _find_two_common_array_index_pairs(array1, array2):
    """Return two index pairs (i, j) such that array1[i] = array2[j]."""
    offset = 0
    index_pairs = _np.empty((2, 2), dtype=_np.int32)
    index_pairs[:, 0] = _find_first_common_array_index_pair_from_position(array1, array2, offset)
    offset = index_pairs[0, 0] + 1  # Next search starts behind found pair
    index_pairs[:, 1] = _find_first_common_array_index_pair_from_position(array1, array2, offset)
    return index_pairs


@_numba.njit()
def _get_shared_vertex_information_for_two_elements(elements, elem0, elem1):
    """
    Return tuple (i, j).

    The tuple has the property elements[i, elem0] == elements[j, elem1]
    """
    i, j = _find_first_common_array_index_pair_from_position(elements[:, elem0], elements[:, elem1])
    return (i, j)


@_numba.njit()
def _get_shared_edge_information_for_two_elements(elements, elem0, elem1):
    """
    Return 2x2 array of int32 indices.

    Each column in the return indices as a pair (i, j) such that
    elements[i, elem0] = elements[j, elem1]

    """
    index_pairs = _find_two_common_array_index_pairs(elements[:, elem0], elements[:, elem1])

    # Ensure that order of indices is the same as Bempp 3

    if index_pairs[1, 1] < index_pairs[1, 0]:
        for i in range(2):
            tmp = index_pairs[i, 0]
            index_pairs[i, 0] = index_pairs[i, 1]
            index_pairs[i, 1] = tmp

    return index_pairs


@_numba.njit()
def _find_vertex_adjacency(elements, test_indices, trial_indices):
    """
    Return for element pairs the vertex adjacency.

    The return array vertex_adjacency has 4 rows, such that for index j
    the element vertex_adjacency[0, j] is connected with
    vertex_adjacency[1, j] via the vertex vertex_adjacency[2, j] in
    the first element and the vertex vertex_adjacency[3, j] in the
    second element. The vertex numbers here are local
    numbers (0, 1 or 2).

    """
    number_of_indices = len(test_indices)
    adjacency = _np.zeros((4, number_of_indices), dtype=_np.int32)

    for index in range(number_of_indices):
        # Now find the position of the shared vertex
        test_index = test_indices[index]
        trial_index = trial_indices[index]
        i, j = _get_shared_vertex_information_for_two_elements(elements, test_index, trial_index)
        adjacency[:, index] = (test_index, trial_index, i, j)

    return adjacency


@_numba.njit()
def _find_edge_adjacency(elements, elem0_indices, elem1_indices):
    """
    Return for element pairs the edge adjacency.

    The return array edge_adjacency has 6 rows, such that for index
    j the element edge_adjacency[0, j] is connected with
    edge_adjacency[1, j] via the two vertices edge_adjacency[2:4, j]
    in the first element and the vertices edge_adjacency[4:6, j]
    in the second element. The vertex numbers here are local
    numbers (0, 1 or 2).

    """
    number_of_indices = len(elem0_indices)

    adjacency = _np.zeros((6, number_of_indices), dtype=_np.int32)

    for index in range(number_of_indices):
        elem0 = elem0_indices[index]
        elem1 = elem1_indices[index]
        index_pairs = _get_shared_edge_information_for_two_elements(elements, elem0, elem1)
        adjacency[0, index] = elem0
        adjacency[1, index] = elem1
        adjacency[2:, index] = index_pairs.flatten()

    return adjacency


def _get_element_to_element_vertex_count(element_to_element_matrix):
    """
    Return a tuple of arrays (elements1, elements2, nvertices).

    The element elements1[i] is connected with elements2[i] via
    nvertices[i] vertices.

    """
    coo_matrix = element_to_element_matrix.tocoo()
    elements1 = coo_matrix.row
    elements2 = coo_matrix.col
    nvertices = coo_matrix.data

    return (elements1, elements2, nvertices)


def _element_filter(elements1, elements2, nvertices, filter_type):
    """
    Return element pairs according to a filter condition.

    Takes an array (elements1, elements2, nvertices)
    such that elements1[i] and elements2[i] are connected
    via nvertices[i] vertices and returns a tuple
    (new_elem1, new_elem2) of all element pairs connected via
    vertices (filter_type=VERTICES) or edges (filter_type=EDGES).

    """
    # Elements connected via edges share two vertices
    filtered_indices = _np.argwhere(nvertices == filter_type).flatten()
    return (elements1[filtered_indices], elements2[filtered_indices])


@_numba.njit()
def _sort_values(val1, val2):
    """Return a tuple with the input values sorted."""
    if val1 > val2:
        val1, val2 = val2, val1
    return val1, val2


@_numba.njit()
def _vertices_from_edge_index(element, local_index):
    """
    Return the vertices associated with an edge.

    Element is 3-tupel with the vertex indices.
    Sorts the returned vertices in ascending order.

    """
    vertex0, vertex1 = element[_EDGE_LOCAL[local_index]]
    return _sort_values(vertex0, vertex1)


def grid_from_segments(grid, segments):
    """Return new grid from segments of existing grid."""
    element_in_new_grid = _np.full(grid.number_of_elements, False)

    for elem in range(grid.number_of_elements):
        if grid.domain_indices[elem] in segments:
            element_in_new_grid[elem] = True
    new_elements = grid.elements[:, element_in_new_grid]
    new_domain_indices = grid.domain_indices[element_in_new_grid]
    vertex_indices = list(set(new_elements.ravel()))
    new_vertices = grid.vertices[:, vertex_indices]
    new_vertex_map = -_np.ones(grid.number_of_vertices, dtype=_np.int_)
    new_vertex_map[vertex_indices] = _np.arange(len(vertex_indices))
    new_elements = new_vertex_map[new_elements.ravel()].reshape(3, -1)

    return Grid(new_vertices, new_elements, new_domain_indices)


@_numba.njit
def _create_barycentric_connectivity_array(vertices, elements, element_edges, edges, number_of_edges):
    """Return the vertices and elements of refined barycentric grid."""
    number_of_vertices = vertices.shape[1]
    number_of_elements = elements.shape[1]
    new_number_of_vertices = number_of_vertices + number_of_elements + number_of_edges
    new_vertices = _np.empty((3, new_number_of_vertices), dtype=_np.float64)
    new_elements = _np.empty((3, 6 * number_of_elements), dtype=_np.float64)

    edge_to_vertex = -_np.ones(number_of_edges)

    new_vertices[:, :number_of_vertices] = vertices

    local_vertex_ids = _np.empty(3, dtype=_np.uint32)

    for index in range(number_of_elements):
        # Create barycentric mid-point
        new_vertices[:, number_of_vertices] = 1.0 / 3 * _np.sum(vertices[:, elements[:, index]], axis=1)
        midpoint_index = number_of_vertices
        number_of_vertices += 1
        for local_index in range(3):
            edge_index = element_edges[local_index, index]
            if edge_to_vertex[edge_index] > -1:
                # Vertex already created
                local_vertex_ids[local_index] = edge_to_vertex[edge_index]
            else:
                # Vertex needs to be created
                new_vertices[:, number_of_vertices] = 0.5 * _np.sum(vertices[:, edges[:, edge_index]], axis=1)
                local_vertex_ids[local_index] = number_of_vertices
                edge_to_vertex[edge_index] = number_of_vertices
                number_of_vertices += 1
            # Have created all necessary vertices. Now create the elements.
            # New barycentric elements are created in anti-clockwise order
            # starting with the triangle at the first vertex of the triangle
            # and sharing a segment with the edge 0. The second triangle is
            # along the same edge, but adjacent to vertex 1, and so on.

        new_elements[0, 6 * index + 0] = elements[0, index]
        new_elements[1, 6 * index + 0] = local_vertex_ids[0]
        new_elements[2, 6 * index + 0] = midpoint_index

        new_elements[0, 6 * index + 1] = elements[1, index]
        new_elements[1, 6 * index + 1] = midpoint_index
        new_elements[2, 6 * index + 1] = local_vertex_ids[0]

        new_elements[0, 6 * index + 2] = elements[1, index]
        new_elements[1, 6 * index + 2] = local_vertex_ids[2]
        new_elements[2, 6 * index + 2] = midpoint_index

        new_elements[0, 6 * index + 3] = elements[2, index]
        new_elements[1, 6 * index + 3] = midpoint_index
        new_elements[2, 6 * index + 3] = local_vertex_ids[2]

        new_elements[0, 6 * index + 4] = elements[2, index]
        new_elements[1, 6 * index + 4] = local_vertex_ids[1]
        new_elements[2, 6 * index + 4] = midpoint_index

        new_elements[0, 6 * index + 5] = elements[0, index]
        new_elements[1, 6 * index + 5] = midpoint_index
        new_elements[2, 6 * index + 5] = local_vertex_ids[1]

    return new_vertices, new_elements


def barycentric_refinement(grid):
    """Return the barycentric refinement of a given grid."""
    new_vertices, new_elements = _create_barycentric_connectivity_array(
        grid.vertices,
        grid.elements,
        grid.element_edges,
        grid.edges,
        grid.number_of_edges,
    )

    return Grid(new_vertices, new_elements, _np.repeat(grid.domain_indices, 6), scatter=False)


def union(grids, domain_indices=None, swapped_normals=None, normalize_domain_indices=True):
    """
    Return the union of a given list of grids.

    Parameters
    ----------
    grids: list
        A list of grid objects.
    domain_indices : list
        Attach a list of domain indices to the new
        grid such that grid[j] received the domain
        index domain_indices[j]
    swapped_normals : list of boolean
        A list of the form [False, True, ...],
        that specifies for each grid if the normals
        should be swapped (True) or not (False). This
        is helpful if one grid is defined to be inside
        another grid.
    normalize_domain_indices : bool
        Normalize the resulting grid with domain unique indices
        between 0 and N-1, where N is the total number
        of domains across all grids, when
        domain_indices has not been provided.

    This method returns a new grid object, which is
    the union of the input grid objects.

    """
    from bempp_cl.api.grid.grid import Grid

    vertex_offset = 0
    element_offset = 0

    vertex_count = sum([grid.number_of_vertices for grid in grids])
    element_count = sum([grid.number_of_elements for grid in grids])

    vertices = _np.empty((3, vertex_count), dtype="float64")
    elements = _np.empty((3, element_count), dtype="uint32")
    all_domain_indices = _np.empty(element_count, dtype="uint32")

    def normalize_array(arr):
        """Normalize domain indices array."""
        arr = arr - arr.min()
        for i, index in enumerate(_np.unique(arr)[1:], start=1):
            arr[arr == index] = i
        return arr

    if domain_indices is None:
        if not normalize_domain_indices:
            domain_indices = [grids[0].domain_indices]
            for grid in grids[1:]:
                domain_indices.append(domain_indices[-1].max() - grid.domain_indices.min() + 1 + grid.domain_indices)
        else:
            domain_indices = [normalize_array(grids[0].domain_indices)]
            for grid in grids[1:]:
                domain_indices.append(domain_indices[-1].max() + 1 + normalize_array(grid.domain_indices))

    if swapped_normals is None:
        swapped_normals = len(grids) * [False]

    for index, grid in enumerate(grids):
        nelements = grid.number_of_elements
        nvertices = grid.number_of_vertices
        vertices[:, vertex_offset : vertex_offset + nvertices] = grid.vertices
        if swapped_normals[index]:
            current_elements = grid.elements[[0, 2, 1], :]
        else:
            current_elements = grid.elements
        elements[:, element_offset : element_offset + nelements] = current_elements + vertex_offset
        all_domain_indices[element_offset : element_offset + nelements] = domain_indices[index]
        vertex_offset += nvertices
        element_offset += nelements
    return Grid(vertices, elements, all_domain_indices)


def enumerate_vertex_adjacent_elements(grid, support_elements, swapped_normals=None):
    """
    Enumerate in anti-clockwise order all elements adjacent to all vertices in support.

    Returns a list [neighbors_0, neighbors_1, ...], where neighbors_i is a list
    [(elem_index, local_ind1, local_ind2), ...] of tuples, where elem_index is an
    element in the support that as connected with vertex i. local_ind1 and local_ind2 are
    the local indices of the two edges that are adjacent to vertex i. They are sorted in
    anti-clockwise order with respect to the natural normal directions of the elements.
    Moreover, all tuples represent elements in anti-clockwise order.
    """
    if swapped_normals is None:
        swapped_normals = []

    vertex_edges = [[] for _ in range(grid.vertices.shape[1])]

    for element_index in support_elements:
        for local_index, edge_index in enumerate(grid.element_edges[:, element_index]):
            for ind in range(2):
                vertex_edges[grid.edges[ind, edge_index]].append((element_index, local_index))

    for vertex_index, neighbors in enumerate(vertex_edges):
        # First sort by element
        if not neighbors:
            # Continue if empty
            continue
        # Now sort each list so that edges appear in anti-clockwise order according
        # to neighboring edges.

        # Swap the edges in each element so
        # that they have edges in anti-clockwise order
        locally_sorted_neighbors = []
        while neighbors:
            # Take first element in list
            elem1 = neighbors.pop()
            for index, elem2 in enumerate(neighbors):
                # Find index of next list element associated
                # with the same grid element
                if elem2[0] == elem1[0]:
                    neighbors.pop(index)
                    break
            # Check if the two edges in the found element entries
            # are in clockwise or anti-clockwise order.
            # Resort accordingly
            if grid.domain_indices[elem1[0]] in swapped_normals:
                if elem1[1] == (1 + elem2[1]) % 3:
                    locally_sorted_neighbors.append((elem1[0], elem1[1], elem2[1]))
                else:
                    locally_sorted_neighbors.append((elem1[0], elem2[1], elem1[1]))
            else:
                if elem1[1] == (1 + elem2[1]) % 3:
                    locally_sorted_neighbors.append((elem1[0], elem2[1], elem1[1]))
                else:
                    locally_sorted_neighbors.append((elem1[0], elem1[1], elem2[1]))

        # locally sorted neighbors now has triplets (elem_index, local_ind1, local_ind2) of
        # one element index and two associated edge indices that are anti-clockwise sorted.
        sorted_neighbors = []
        sorted_neighbors.append(locally_sorted_neighbors.pop())
        while locally_sorted_neighbors:
            found = False
            for index, elem in enumerate(locally_sorted_neighbors):
                # Check if element is successor of last element in sorted list
                last = sorted_neighbors[-1]
                first = sorted_neighbors[0]
                if grid.data().element_edges[elem[1], elem[0]] == grid.data().element_edges[last[2], last[0]]:
                    locally_sorted_neighbors.pop(index)
                    found = True
                    sorted_neighbors.append(elem)
                    break
                if grid.data().element_edges[elem[2], elem[0]] == grid.data().element_edges[first[1], first[0]]:
                    locally_sorted_neighbors.pop(index)
                    found = True
                    sorted_neighbors.insert(0, elem)
                    break
            if not found:
                raise Exception("Two elements seem to be connected only by a vertex, not by an edge.")

        vertex_edges[vertex_index] = sorted_neighbors

    return vertex_edges


@_numba.njit
def _numba_enumerate_edges(elements, edge_tuple_to_index):
    """
    Enumerate all edges in a given grid.

    Assigns a tuple (edges, element_edges) to
    self._edges and self._element_edges.
    element_edges is an array a such that a[i, j] is the
    index of the ith edge in the jth elements, and edges
    is a 2 x nedges array such that the jth column stores the
    two nodes associated with the jth edge.

    """
    edges = []

    number_of_elements = elements.shape[1]
    element_edges = _np.zeros((3, number_of_elements), dtype=_np.int32)

    number_of_edges = 0

    for elem_index in range(number_of_elements):
        elem = elements[:, elem_index]
        for local_index in range(3):
            edge_tuple = _vertices_from_edge_index(elem, local_index)
            if edge_tuple not in edge_tuple_to_index:
                edge_index = number_of_edges
                edge_tuple_to_index[edge_tuple] = edge_index
                edges.append(edge_tuple)
                number_of_edges += 1
            else:
                edge_index = edge_tuple_to_index[edge_tuple]
            element_edges[local_index, elem_index] = edge_index

    return _np.array(edges, dtype=_np.int32).T, element_edges


def _grid_scatter_worker(grid_id, array_proxies):
    """Assign a new grid on the worker."""
    from bempp_cl.api.utils import pool
    from bempp_cl.api.grid.grid import Grid
    from bempp_cl.api import log

    vertices, elements, domain_indices = pool.from_buffer(array_proxies)

    # if not pool.has_key(grid_id):
    if grid_id not in pool:
        pool.insert_data(
            grid_id,
            Grid(vertices.copy(), elements.copy(), domain_indices.copy(), grid_id),
        )
        log(f"Copied grid with id {grid_id} to worker {pool.get_id()}", "debug")
    else:
        log(f"Use cached grid with id {grid_id} on worker {pool.get_id()}", "debug")


@_numba.njit
def grid_to_points(grid_data, local_points):
    """
    Map a grid to an array of points.

    Returns a (N, 3) point array that stores the global vertices
    associated with the local points in each triangle.
    Points are stored in consecutive order for each element
    in the support_elements list. Hence, the returned array is of the form
    [ v_1^1, v_2^1, ..., v_M^1, v_1^2, v_2^2, ...], where
    v_i^j is the ith point in the jth element in
    the support_elements list.

    Parameters
    ----------
    grid_data : GridData
        A Bempp GridData object.
    local_points : np.ndarray
        (2, M) array of local coordinates.
    """
    number_of_elements = grid_data.elements.shape[1]
    number_of_points = local_points.shape[1]

    points = _np.empty((number_of_points * number_of_elements, 3), dtype=_np.float64)

    for elem in range(number_of_elements):
        points[number_of_points * elem : number_of_points * (1 + elem), :] = (
            _np.expand_dims(grid_data.vertices[:, grid_data.elements[0, elem]], 1)
            + grid_data.jacobians[elem].dot(local_points)
        ).T
    return points


def _get_barycentric_support(truncate_at_segment_edge, grid, bary_grid, coarse_space):
    coarse_support = _np.zeros(grid.entity_count(0), dtype=_np.bool_)
    coarse_support[coarse_space.support_elements] = True

    if not truncate_at_segment_edge:
        for global_dof_index in range(coarse_space.global_dof_count):
            local_dofs = coarse_space.global2local[global_dof_index]
            edge_index = grid.data().element_edges[local_dofs[0][1], local_dofs[0][0]]
            for v in range(2):
                vertex = grid.data().edges[v, edge_index]
                start = grid.vertex_neighbors.indexptr[vertex]
                end = grid.vertex_neighbors.indexptr[vertex + 1]
                for cell in grid.vertex_neighbors.indices[start:end]:
                    coarse_support[cell] = True

    coarse_support_elements = _np.array([i for i, j in enumerate(coarse_support) if j])
    number_of_support_elements = len(coarse_support_elements)

    bary_support_elements = 6 * _np.repeat(coarse_support_elements, 6) + _np.tile(
        _np.arange(6), number_of_support_elements
    )

    support = _np.zeros(bary_grid.number_of_elements, dtype=_np.bool_)
    support[bary_support_elements] = True

    bary_support_size = len(bary_support_elements)

    return support, bary_support_size, bary_support_elements


def _get_data_multipliers(support, bary_grid, bary_support_elements, swapped_normals, coarse_space, bary_support_size):
    bary_vertex_to_edge = enumerate_vertex_adjacent_elements(bary_grid, bary_support_elements, swapped_normals)

    edge_vectors = bary_grid.vertices[:, bary_grid.edges[0, :]] - bary_grid.vertices[:, bary_grid.edges[1, :]]

    edge_lengths = _np.linalg.norm(edge_vectors, axis=0)

    normal_multipliers = _np.repeat(coarse_space.normal_multipliers, 6)
    local2global = _np.zeros((bary_grid.number_of_elements, 3), dtype="uint32")
    local_multipliers = _np.zeros((bary_grid.number_of_elements, 3), dtype="uint32")

    local2global[support] = _np.arange(3 * bary_support_size).reshape(bary_support_size, 3)

    local_multipliers[support] = 1

    return bary_vertex_to_edge, local2global, edge_lengths, normal_multipliers, local_multipliers


def _get_barycentric_edges_associated_to_vertex(vertex_index, bary_vertex_to_edge, bary_element, bary_grid):
    """Get all the barycentryc vertices of the elements associated to a reference edge."""
    # Get barycentric elements associated to a vertex
    for ind, elem in enumerate(bary_vertex_to_edge[vertex_index]):
        if bary_element == elem[0]:
            break

    # Get all the relevant barycentric edges starting to count above ind
    num_bary_elements = len(bary_vertex_to_edge[vertex_index])
    vertex_edges = []
    for index in range(num_bary_elements):
        elem_edge_pair = bary_vertex_to_edge[vertex_index][(index + ind) % num_bary_elements]
        for n in range(1, 3):
            vertex_edges.append((elem_edge_pair[0], elem_edge_pair[n]))
    edges = []
    for vertex in vertex_edges:
        edges.append(bary_grid.data().element_edges[vertex[1]][vertex[0]])
    set_edges = set(edges)
    sorted_edges = []
    new_vertex_edges = []
    ref_edge = 0

    for el in set_edges:
        # edges.count(el) == 1 means that the edge is associated to one element,
        # therefore, it is a barycentric element on the border of a on open geometry.
        # vertex_edges[edges.index(el)][1] == 1 indicates that the barycentric element
        # on the border is the first one anti-clock-wise.
        if edges.count(el) == 1 and vertex_edges[edges.index(el)][1] == 1:
            sorted_edges.append(el)
            new_vertex_edges.append(vertex_edges[edges.index(el)])

    # If the basis function is on the border, sort the rest of the elements accordingly.
    # The basis functions on the borders do not need to remove the reference edge.
    if len(sorted_edges) > 0:
        new_vertex_edges, sorted_edges = _sort_vertex_edges(
            vertex_edges, edges, set_edges, sorted_edges, new_vertex_edges, 1
        )
        sorted_edges = sorted(set(sorted_edges), key=sorted_edges.index, reverse=True)
        new_vertex_edges = sorted(set(new_vertex_edges), key=new_vertex_edges.index, reverse=True)
        ref_edge = sorted_edges.index(bary_grid.data().element_edges[vertex_edges[0][1]][vertex_edges[0][0]])
    else:
        # If the first element is not on the border, sorting is not necessary
        # and we remove the barycentric edges that lie on the reference edge.
        vertex_edges.pop(0)
        vertex_edges.pop(-1)

    return vertex_edges, sorted_edges, num_bary_elements // 2, ref_edge


def get_vertex_edges(vertex_index, bary_vertex_to_edge, bary_element, bary_grid):
    """Get all the barycentryc vertices of the elements associated to a reference edge."""
    for ind, elem in enumerate(bary_vertex_to_edge[vertex_index]):
        if bary_element == elem[0]:
            break
    # Get all the relevant edges starting to count above
    # ind
    num_bary_elements = len(bary_vertex_to_edge[vertex_index])
    vertex_edges = []
    for index in range(num_bary_elements):
        elem_edge_pair = bary_vertex_to_edge[vertex_index][(index + ind) % num_bary_elements]
        for n in range(1, 3):
            vertex_edges.append((elem_edge_pair[0], elem_edge_pair[n]))
    edges = []
    for vertex in vertex_edges:
        edges.append(bary_grid.data().element_edges[vertex[1]][vertex[0]])
    set_edges = set(edges)
    sorted_edges = []
    new_vertex_edges = []
    ref_edge = 0

    for el in set_edges:
        if edges.count(el) == 1 and vertex_edges[edges.index(el)][1] == 1:
            sorted_edges.append(el)
            new_vertex_edges.append(vertex_edges[edges.index(el)])

    if len(sorted_edges) > 0:
        new_vertex_edges, sorted_edges = _sort_vertex_edges(
            vertex_edges, edges, set_edges, sorted_edges, new_vertex_edges, 1
        )
        sorted_edges = sorted(set(sorted_edges), key=sorted_edges.index, reverse=True)
        new_vertex_edges = sorted(set(new_vertex_edges), key=new_vertex_edges.index, reverse=True)
        ref_edge = sorted_edges.index(bary_grid.data().element_edges[vertex_edges[0][1]][vertex_edges[0][0]])
    else:
        # We do not want the reference edge part of this list
        vertex_edges.pop(0)
        vertex_edges.pop(-1)

    return vertex_edges, sorted_edges, num_bary_elements // 2, ref_edge


def _sort_vertex_edges(vertex_edges, edges, set_edges, sorted_edges, new_vertex_edges, edge_number):
    """Sort edges anti clock-wise when the basis function lies on the border of a geometry."""
    if edge_number == 1:
        for element in vertex_edges:
            if element[0] == new_vertex_edges[-1][0] and element[1] == 0:
                sorted_edges.append(edges[vertex_edges.index((new_vertex_edges[-1][0], 0))])
                new_vertex_edges.append(element)
    else:
        indexes = [ind for ind, el in enumerate(edges) if el == sorted_edges[-1]]
        if vertex_edges[indexes[0]][1] == 1:
            sorted_edges.append(sorted_edges[-1])
            new_vertex_edges.append(vertex_edges[indexes[0]])
        elif vertex_edges[indexes[1]][1] == 1:
            sorted_edges.append(sorted_edges[-1])
            new_vertex_edges.append(vertex_edges[indexes[1]])
    if len(new_vertex_edges) < len(vertex_edges):
        return _sort_vertex_edges(vertex_edges, edges, set_edges, sorted_edges, new_vertex_edges, 1 - edge_number)
    return new_vertex_edges, sorted_edges


def _get_bary_coefficients(
    edge_lengths,
    vertex_edges1,
    vertex_edges2,
    sorted_edges1,
    sorted_edges2,
    bary_grid,
    local2global,
    nc1,
    nc2,
    global_dof_index,
    ref_edge1,
    ref_edge2,
):
    """Obtain the edge coefficients associated to the barycentric edges of a grid."""
    values = []
    bary_dofs = []
    coarse_dofs = []

    # Check whether any of the poles of the BC basis function is located on the border of a geometry.
    border_edges1 = _check_if_border_edges(vertex_edges1, bary_grid)
    border_edges2 = _check_if_border_edges(vertex_edges2, bary_grid)

    if border_edges1 and not border_edges2:
        aux_values, aux_bary_dofs, aux_coarse_dofs = _border_barycentric_edges_coefficients(
            edge_lengths, vertex_edges1, sorted_edges1, bary_grid, local2global, -1.0, nc1, global_dof_index, ref_edge1
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
        aux_values, aux_bary_dofs, aux_coarse_dofs = _interior_barycentric_edges_coefficients(
            edge_lengths, vertex_edges2, bary_grid, local2global, 1.0, nc2, global_dof_index
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
    elif not border_edges1 and border_edges2:
        aux_values, aux_bary_dofs, aux_coarse_dofs = _interior_barycentric_edges_coefficients(
            edge_lengths, vertex_edges1, bary_grid, local2global, -1.0, nc1, global_dof_index
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
        aux_values, aux_bary_dofs, aux_coarse_dofs = _border_barycentric_edges_coefficients(
            edge_lengths, vertex_edges2, sorted_edges2, bary_grid, local2global, 1.0, nc2, global_dof_index, ref_edge2
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
    elif border_edges1 and border_edges2:
        aux_values, aux_bary_dofs, aux_coarse_dofs = _border_barycentric_edges_coefficients(
            edge_lengths, vertex_edges1, sorted_edges1, bary_grid, local2global, -1.0, nc1, global_dof_index, ref_edge1
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
        aux_values, aux_bary_dofs, aux_coarse_dofs = _border_barycentric_edges_coefficients(
            edge_lengths, vertex_edges2, sorted_edges2, bary_grid, local2global, 1.0, nc2, global_dof_index, ref_edge2
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
    else:
        aux_values, aux_bary_dofs, aux_coarse_dofs = _interior_barycentric_edges_coefficients(
            edge_lengths, vertex_edges1, bary_grid, local2global, -1.0, nc1, global_dof_index
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
        aux_values, aux_bary_dofs, aux_coarse_dofs = _interior_barycentric_edges_coefficients(
            edge_lengths, vertex_edges2, bary_grid, local2global, 1.0, nc2, global_dof_index
        )
        values += aux_values
        bary_dofs += aux_bary_dofs
        coarse_dofs += aux_coarse_dofs
    return values, bary_dofs, coarse_dofs


def _check_if_border_edges(vertex_edges, bary_grid):
    """Check if an edge belongs to the border of the grid."""
    border_edges = False
    for edge in vertex_edges:
        elem_index, local_edge_index = edge[:]
        edge_index = bary_grid.data().element_edges[local_edge_index, elem_index]
        neighbors = bary_grid.edge_neighbors[edge_index]
        if len(neighbors) == 1:
            border_edges = True
            break
    return border_edges


def _border_barycentric_edges_coefficients(
    edge_lengths, vertex_edges, sorted_edges, bary_grid, local2global, sign, nc, global_dof_index, ref_edge
):
    """Calculate barycentric edge coefficients associated to an interior vertex of the grid."""
    values = []
    bary_dofs = []
    coarse_dofs = []
    count = 0
    signs = sign * _np.array([-1.0, 1.0])
    for edge in vertex_edges:
        elem_index, local_edge_index = edge[:]
        edge_length = edge_lengths[bary_grid.data().element_edges[local_edge_index, elem_index]]
        count = sorted_edges.index(bary_grid.data().element_edges[local_edge_index, elem_index])

        bary_dofs.append(local2global[elem_index, local_edge_index])
        coarse_dofs.append(global_dof_index)
        if count < ref_edge:
            values.append(signs[local_edge_index] * (1 - nc) / (nc * edge_length))
        elif count == ref_edge:
            values.append(signs[local_edge_index] * (2 - nc) / (2 * nc * edge_length))
        else:
            values.append(signs[local_edge_index] * 1.0 / (nc * edge_length))
    return values, bary_dofs, coarse_dofs


def _interior_barycentric_edges_coefficients(
    edge_lengths, vertex_edges, bary_grid, local2global, sign, nc, global_dof_index
):
    """Calculate barycentric edge coefficients associated to a vertex that belongs to the border of the grid.

    (eg: vertex on the edge of a screen).
    """
    values = []
    bary_dofs = []
    coarse_dofs = []
    count = 0
    for index, edge in enumerate(vertex_edges):
        if index % 2 == 0:
            count += 1
        elem_index, local_edge_index = edge[:]
        edge_length = edge_lengths[bary_grid.data().element_edges[local_edge_index, elem_index]]
        bary_dofs.append(local2global[elem_index, local_edge_index])
        coarse_dofs.append(global_dof_index)
        values.append(sign * (nc - count) / (2 * nc * edge_length))
        sign *= -1
    return values, bary_dofs, coarse_dofs


def _get_coefficients_reference_edge(
    edge_lengths,
    bary_grid,
    local2global,
    global_dof_index,
    bary_upper_minus,
    bary_upper_plus,
    bary_lower_minus,
    bary_lower_plus,
):
    """Calculate upper and lower coefficients of the barycentric edges associated to a reference edge."""
    values = []
    bary_dofs = []
    coarse_dofs = []

    coarse_dofs.append(global_dof_index)
    coarse_dofs.append(global_dof_index)
    coarse_dofs.append(global_dof_index)
    coarse_dofs.append(global_dof_index)

    edge_length_upper = edge_lengths[bary_grid.data().element_edges[2, bary_upper_minus]]
    edge_length_lower = edge_lengths[bary_grid.data().element_edges[2, bary_lower_minus]]

    bary_dofs.append(local2global[bary_upper_minus, 2])
    bary_dofs.append(local2global[bary_upper_plus, 2])
    bary_dofs.append(local2global[bary_lower_minus, 2])
    bary_dofs.append(local2global[bary_lower_plus, 2])

    values.append(1.0 / (2 * edge_length_upper))
    values.append(-1.0 / (2 * edge_length_upper))
    values.append(-1.0 / (2 * edge_length_lower))
    values.append(1.0 / (2 * edge_length_lower))

    return values, bary_dofs, coarse_dofs
