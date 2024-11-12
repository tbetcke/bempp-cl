"""Interface to DOLFIN for FEM-BEM coupling."""

import dolfin as _dolfin


def boundary_grid_from_fenics_mesh(fenics_mesh):
    """
    Create a Bempp boundary grid from a FEniCS Mesh.

    Return the Bempp grid and a map from the node numberings of the FEniCS
    mesh to the node numbers of the boundary grid.
    """
    import bempp_cl.api
    import numpy as np

    boundary_mesh = _dolfin.BoundaryMesh(fenics_mesh, "exterior", False)
    bm_coords = boundary_mesh.coordinates()
    bm_cells = boundary_mesh.cells()
    bm_nodes = boundary_mesh.entity_map(0).array().astype(np.int64)
    bempp_boundary_grid = bempp_cl.api.Grid(bm_coords.transpose(), bm_cells.transpose())
    return bempp_boundary_grid, bm_nodes


def fenics_to_bempp_trace_data(fenics_space):
    """Return tuple (space,trace_matrix)."""
    family, degree = fenics_space_info(fenics_space)

    if family == "Lagrange":
        if degree == 1:
            return p1_trace(fenics_space)
        else:
            raise NotImplementedError()
    elif family == "Nedelec 1st kind H(curl)":
        if degree == 1:
            return nc1_tangential_trace(fenics_space)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def fenics_space_info(fenics_space):
    """Return tuple (family,degree) containing information about a FEniCS space."""
    element = fenics_space.ufl_element()
    family = element.family()
    degree = element.degree()
    return (family, degree)


class FenicsOperator(object):
    """Wrap a FEniCS Operator into a Bempp operator."""

    def __init__(self, fenics_weak_form):
        """Construct an operator from a weak form in FEniCS."""
        self._fenics_weak_form = fenics_weak_form
        self._sparse_mat = None

    def weak_form(self):
        """Return the weak form."""
        from bempp_cl.api.assembly.discrete_boundary_operator import (
            SparseDiscreteBoundaryOperator,
        )
        from dolfin import as_backend_type, assemble, parameters
        from scipy.sparse import csr_matrix

        if self._sparse_mat is not None:
            return SparseDiscreteBoundaryOperator(self._sparse_mat)

        backend = parameters["linear_algebra_backend"]
        if backend != "PETSc":
            raise ValueError("Only the PETSc linear algebra backend is supported.")
        mat = as_backend_type(assemble(self._fenics_weak_form)).mat()
        (indptr, indices, data) = mat.getValuesCSR()
        self._sparse_mat = csr_matrix((data, indices, indptr), shape=mat.size)

        return SparseDiscreteBoundaryOperator(self._sparse_mat)


# pylint: disable=too-many-locals
def p1_trace(fenics_space):
    """
    Return the P1 trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a FEniCS function to its boundary
    trace coefficients in the corresponding Bempp space.

    """
    import bempp_cl.api
    from scipy.sparse import coo_matrix
    import numpy as np

    family, degree = fenics_space_info(fenics_space)
    if not (family == "Lagrange" and degree == 1):
        raise ValueError("fenics_space must be a p1 Lagrange space")

    fenics_mesh = fenics_space.mesh()
    bempp_boundary_grid, bm_nodes = boundary_grid_from_fenics_mesh(fenics_mesh)

    # First get trace space
    space = bempp_cl.api.function_space(bempp_boundary_grid, "P", 1)

    # FEniCS vertices to boundary dofs
    b_vertices_from_vertices = coo_matrix(
        (np.ones(len(bm_nodes)), (np.arange(len(bm_nodes)), bm_nodes)),
        shape=(len(bm_nodes), fenics_mesh.num_vertices()),
        dtype="float64",
    ).tocsc()

    # Finally FEniCS dofs to vertices.
    vertices_from_fenics_dofs = coo_matrix(
        (
            np.ones(fenics_mesh.num_vertices()),
            (
                _dolfin.dof_to_vertex_map(fenics_space),
                np.arange(fenics_mesh.num_vertices()),
            ),
        ),
        shape=(fenics_mesh.num_vertices(), fenics_mesh.num_vertices()),
        dtype="float64",
    ).tocsc()

    # Get trace matrix by multiplication
    trace_matrix = b_vertices_from_vertices @ vertices_from_fenics_dofs

    # Now return everything
    return space, trace_matrix


# pylint: disable=too-many-locals
def nc1_tangential_trace(fenics_space):
    """
    Return the NC1 (twisted) tangential trace operator.

    This function returns a pair (space, trace_matrix),
    where space is a Bempp space object and trace_matrix is the corresponding
    matrix that maps the coefficients of a FEniCS function to its boundary
    trace coefficients in the corresponding Bempp space.

    """
    import bempp_cl.api
    from scipy.sparse import coo_matrix
    import numpy as np

    family, degree = fenics_space_info(fenics_space)
    if not (family == "Nedelec 1st kind H(curl)" and degree == 1):
        raise ValueError("fenics_space must be an order 1 Nedelec 1st kind H(curl) space")

    fenics_mesh = fenics_space.mesh()
    bempp_boundary_grid, bm_nodes = boundary_grid_from_fenics_mesh(fenics_mesh)
    fenics_dim = fenics_space.dim()

    # First get trace space
    space = bempp_cl.api.function_space(bempp_boundary_grid, "RWG", 0)

    # Now compute the mapping from Bempp dofs to FEniCS dofs
    # Overall plan:
    #   bempp_dofs <- bd_vertex_pairs <- all_vertex_pairs <- all_edges <- fenics_dofs

    # First the Bempp dofs to the boundary edges
    #   bempp_dofs <- bd_vertex_pairs

    grid = space.grid
    edge_count = space.global_dof_count
    face_count = grid.entity_count(0)

    dof_to_vertices_map = np.zeros((edge_count, 2), dtype=np.int64)
    dof_to_face_map = [None] * edge_count

    for element in grid.entity_iterator(0):
        g_d = space.local2global[element.index]
        vs = [v.index for v in element.sub_entity_iterator(2)]
        for i, e in enumerate([(0, 1), (0, 2), (1, 2)]):
            dof_to_vertices_map[g_d[i], 0] = vs[e[0]]
            dof_to_vertices_map[g_d[i], 1] = vs[e[1]]

            dof_to_face_map[g_d[i]] = (element, i)

    # Now the boundary triangle edges to the tetrahedron faces
    #   bd_vertex_pairs <- all_vertex_pairs
    # is bm_nodes

    #   all_vertex_pairs <- all_edges
    all_vertices_to_all_edges = {}
    for edge in _dolfin.entities(fenics_mesh, 1):
        i = edge.index()
        ent = edge.entities(0)
        all_vertices_to_all_edges[(max(ent), min(ent))] = i

    # Make matrix bempp_dofs <- all_edges
    bempp_dofs_to_all_edges_map = np.zeros(space.global_dof_count, dtype=np.int64)
    for bd_e, v in enumerate(dof_to_vertices_map):
        v2 = bm_nodes[v]
        v2 = (max(v2), min(v2))
        all_e = all_vertices_to_all_edges[v2]
        bempp_dofs_to_all_edges_map[bd_e] = all_e

    bempp_dofs_from_all_edges = coo_matrix(
        (
            np.ones(space.global_dof_count),
            (np.arange(space.global_dof_count), bempp_dofs_to_all_edges_map),
        ),
        shape=(space.global_dof_count, fenics_dim),
        dtype="float64",
    ).tocsc()

    # Finally the edges to FEniCS dofs
    #   all_edges <- fenics_dofs
    dofmap = fenics_space.dofmap()
    dof_to_edge_map = np.zeros(fenics_dim, dtype=np.int64)
    for cell in _dolfin.cells(fenics_mesh):
        c_d = dofmap.cell_dofs(cell.index())
        c_e = cell.entities(1)
        for d, e in zip(c_d, c_e):
            dof_to_edge_map[d] = e

    all_edges_from_fenics_dofs = coo_matrix(
        (np.ones(fenics_dim), (dof_to_edge_map, np.arange(fenics_dim))),
        shape=(fenics_dim, fenics_dim),
        dtype="float64",
    ).tocsc()

    # Get trace matrix by multiplication
    #   bempp_dofs <- bd_edges <- all_edges <- fenics_dofs
    trace_matrix = bempp_dofs_from_all_edges * all_edges_from_fenics_dofs

    # Now sort out directions
    v_list = list(space.grid.entity_iterator(2))
    local_coords = [
        np.array([[0.5], [0.0]]),
        np.array([[0.0], [0.5]]),
        np.array([[0.5], [0.5]]),
    ]

    # Build a map from Bempp triangles to FEniCS edges
    # aim: bempp_triangles -> all_edge_triplets -> fenics_tetrahedra
    # First: bempp Triangles -> all edge triplets

    triangles_to_edge_triples = [None] * face_count
    for face in grid.entity_iterator(0):
        es = [e.index for e in face.sub_entity_iterator(2)]
        all_es = [bm_nodes[e] for e in es]
        all_es.sort()
        triangles_to_edge_triples[face.index] = tuple(all_es)
    edge_triples_to_tetrahedra = {}
    for tetra in _dolfin.cells(fenics_mesh):
        all_es = [v.index() for v in _dolfin.vertices(tetra)]
        all_es.sort()
        edge_triples_to_tetrahedra[(all_es[0], all_es[1], all_es[2])] = tetra
        edge_triples_to_tetrahedra[(all_es[0], all_es[1], all_es[3])] = tetra
        edge_triples_to_tetrahedra[(all_es[0], all_es[2], all_es[3])] = tetra
        edge_triples_to_tetrahedra[(all_es[1], all_es[2], all_es[3])] = tetra
    triangles_to_tetrahedra = [edge_triples_to_tetrahedra[triple] for triple in triangles_to_edge_triples]

    # return None, None
    fenics_fun = _dolfin.Function(fenics_space)
    non_z = trace_matrix.nonzero()
    for i, j in zip(non_z[0], non_z[1]):
        fenics_fun.vector()[:] = [1.0 if k == j else 0.0 for k in range(fenics_dim)]
        bempp_fun = bempp_cl.api.GridFunction(
            space,
            coefficients=[1.0 if k == i else 0.0 for k in range(space.global_dof_count)],
        )

        v1 = v_list[dof_to_vertices_map[i][0]]
        v2 = v_list[dof_to_vertices_map[i][1]]
        midpoint = (v1.geometry.corners + v2.geometry.corners) / 2

        face = dof_to_face_map[i]
        fenics_values = np.zeros(3)
        fenics_fun.eval_cell(fenics_values, midpoint.T[0], triangles_to_tetrahedra[face[0].index])

        bempp_values = bempp_fun.evaluate(face[0].index, local_coords[face[1]])
        normal = face[0].geometry.normal
        cross = np.cross(fenics_values, normal)
        k = np.argmax(np.abs(cross))

        trace_matrix[i, j] = cross[k] / bempp_values[k, 0]

    # Now return everything
    return space, trace_matrix
