"""Implementation of an octree in Python."""

import numpy as _np
import numba as _numba


@_numba.experimental.jitclass(
    [
        ("_lbound", _numba.float64[:]),
        ("_ubound", _numba.float64[:]),
        ("_maximum_level", _numba.int32),
        ("_diameter", _numba.float64[:]),
        ("_vertices", _numba.float64[:, :]),
        ("_sorted_indices", _numba.uint32[:]),
        ("_leaf_nodes", _numba.uint32[:]),
        ("_leaf_nodes_index_ptr", _numba.uint32[:]),
        ("_level_nodes", _numba.uint32[:]),
        ("_level_nodes_index_ptr", _numba.uint32[:]),
        ("_near_field_nodes", _numba.int32[:]),
    ]
)
class Octree(object):
    """Data structure for handling Octrees."""

    def __init__(self, lbound, ubound, maximum_level, vertices):
        """
        Initialize an Octree.

        Parameters
        ----------
        lbound : np.ndarray
            Numpy array of size (3, ) that specifies the lower
            bound of the Octree.
        ubound : np.ndarray
            Numpy array of size (3, ) that specifies the upper
            bound of the Octree.
        maximum_level : integer
            The maximum level of the Octree.
        vertices : np.ndarray
            An (3, N) float64 array of N vertices
        """
        self._lbound = lbound
        self._ubound = ubound
        self._maximum_level = maximum_level
        self._diameter = ubound - lbound
        self._vertices = vertices
        self._assign_nodes(vertices)
        self._compute_nearfields()

    @property
    def diameter(self):
        """Return diameter of the Octree in each dimension."""
        return self._diameter

    @property
    def lower_bound(self):
        """Return lower bound of Octree in each dimension."""
        return self._lbound

    @property
    def upper_bound(self):
        """Return upper bound of Octree in each dimension."""
        return self._ubound

    @property
    def maximum_level(self):
        """Return the maximum level."""
        return self._maximum_level

    @property
    def vertices(self):
        """Return the vertices."""
        return self._vertices

    @property
    def non_empty_leaf_nodes(self):
        """Return the non-empty leaf nodes."""
        return self._leaf_nodes

    @property
    def sorted_indices(self):
        """Return the indices sorted by leaf node."""
        return self._sorted_indices

    @property
    def leaf_nodes_ptr(self):
        """
        Return the index pointers for the leaf node elements.

        Returns an array index_ptr, such that the indices
        from the jth non-empty leaf node can be obtained
        by self.sorted_indices[index_ptr[j]:index_ptr[j+1]]
        and the associated node index through
        self._non_empty_leaf_nodes[j].
        """
        return self._leaf_nodes_index_ptr

    @property
    def non_empty_nodes_by_level(self):
        """Return the non-empty nodes by level."""
        return self._level_nodes

    @property
    def non_empty_nodes_ptr(self):
        """
        Return the index pointers for non-empty nodes by level.

        Returns an array index_ptr, such that the non-empty nodes
        of the jth level are given by
        self.non_empty_nodes_by_level[index_ptr[j]:index_ptr[j+1]].
        """
        return self._level_nodes_index_ptr

    @property
    def near_field_nodes(self):
        """Return near field nodes."""
        return self._near_field_nodes

    @property
    def near_field_nodes_ptr(self):
        """
        Return an index ptr to the near field nodes.

        Returns an array index_ptr, such that
        self.near_field_nodes[index_ptr[j]:index_ptr[j+1]]
        contains the near field nodes for all nodes in level j.
        There are 27 entries for each near field node. The
        sequence of nodes is the same as for the array
        non_empty_nodes_by_level.
        """
        return _np.uint32(27) * self._level_nodes_index_ptr

    def parent(self, node_index):
        """Return the parent index of a node."""
        return node_index >> 3

    def children(self, node_index):
        """Return an iterator over the child indices."""
        first = node_index << 3
        last = 7 + (node_index << 3)
        return list(range(first, 1 + last))

    def nodes_per_side(self, level):
        """Return number of nodes along each dimension."""
        return 1 << level

    def nodes_per_level(self, level):
        """Return the number of nodes in a given level."""
        return 1 << 3 * level

    def leaf_containing_point(self, point):
        """Return the Morton index of a node containing the point."""
        leaf_size = self.nodes_per_side(self.maximum_level)

        fractions = ((point - self.lower_bound) / self.diameter) * leaf_size

        indices = _np.fmin(_np.fmax(0, fractions).astype(_np.int32), leaf_size - 1)

        return morton(indices)

    def node_bounds(self, morton_index, level):
        """
        Return the lower/upper bound of a node by Morton index.

        The method returns a tuple (lbound, ubound) which define
        the lower and upper corners of a node given by its Morton index.
        """
        indices = _np.array(de_morton(morton_index))
        nnodes_along_dimension = self.nodes_per_side(level)

        node_size = self.diameter / nnodes_along_dimension
        lbound = self.lower_bound + indices * node_size
        ubound = self.lower_bound + (1 + indices) * node_size
        return (lbound, ubound)

    def neighbors(self, node_index, level):
        """Return a list of indices of the neighbors of a node."""
        return _neighbors(node_index, level)

    def node_diameter(self, level):
        """Return node diameter in a given level."""
        return self.diameter / (1.0 * self.nodes_per_side(self.maximum_level))

    def _assign_nodes(self, vertices):
        """Compute leaf-nodes and parents."""
        nvertices = vertices.shape[1]
        node_indices = _np.empty(nvertices, dtype=_np.uint32)

        for index in range(nvertices):
            node_indices[index] = self.leaf_containing_point(vertices[:, index])

        self._sorted_indices = _np.argsort(node_indices).astype(_np.uint32)
        index_ptr = []
        nodes = []

        tmp = -1

        for i, index in enumerate(self._sorted_indices):
            if tmp != node_indices[index]:
                index_ptr.append(i)
                nodes.append(node_indices[index])
                tmp = node_indices[index]

        index_ptr.append(len(self._sorted_indices))
        self._leaf_nodes = _np.array(nodes, dtype=_np.uint32)
        self._leaf_nodes_index_ptr = _np.array(index_ptr, dtype=_np.uint32)

        # Now compute the parents

        self._level_nodes = self._leaf_nodes
        self._level_nodes_index_ptr = _np.zeros(self.maximum_level + 2, _np.uint32)
        nnodes = _np.empty(1 + self._maximum_level, _np.uint32)
        nnodes[self.maximum_level] = len(self._leaf_nodes)
        current_level = self._leaf_nodes
        current_nodes = []

        for index in range(1, self.maximum_level + 1):
            current_nodes.clear()
            for node_index in current_level:
                current_nodes.append(self.parent(node_index))
            current_level = _make_unique(_np.array(current_nodes, dtype=_np.uint32))
            self._level_nodes = _np.concatenate((current_level, self._level_nodes))
            nnodes[self.maximum_level - index] = len(current_level)

        tmp = 0

        for index in range(0, 1 + self.maximum_level):
            self._level_nodes_index_ptr[index] = tmp
            tmp += nnodes[index]
        self._level_nodes_index_ptr[1 + self.maximum_level] = tmp

    def _compute_nearfields(self):
        """
        Compute near fields of all non empty nodes.

        Each node can have at most 27 near field nodes (including
        the node itself). If a near field node does not exist or is empty
        then the value -1 is stored, otherwise the node number.
        """
        self._near_field_nodes = _np.empty(27 * len(self._level_nodes), _np.int32)

        count = 0

        for level_index in range(self.maximum_level + 1):
            sides = 1 << level_index
            level_nodes = self.non_empty_nodes_by_level[
                self.non_empty_nodes_ptr[level_index] : self.non_empty_nodes_ptr[level_index + 1]
            ]
            for node_index in level_nodes:
                ind1, ind2, ind3 = de_morton(node_index)

                for i in range(-1, 2):
                    for j in range(-1, 2):
                        for k in range(-1, 2):
                            if _in_range(ind1 + i, ind2 + j, ind3 + k, sides):
                                morton_index = morton((ind1 + i, ind2 + j, ind3 + k))
                                if _np.any(level_nodes == morton_index):
                                    self._near_field_nodes[count] = morton_index
                                else:
                                    self._near_field_nodes[count] = -1
                            else:
                                self._near_field_nodes[count] = -1
                            count += 1

    def _compute_interaction_list(self):
        """Compute the interaction list for each non empty node."""
        pass


@_numba.njit
def _make_unique(ar):
    """Find unique elements.

    An implementation of Numpy unique for Numba.
    """
    sorted_ar = _np.sort(ar)
    unique_lst = [sorted_ar[0]]
    for elem in sorted_ar:
        if elem != unique_lst[-1]:
            unique_lst.append(elem)
    return _np.array(unique_lst, dtype=ar.dtype)


@_numba.njit(cache=True)
def _in_range(n1, n2, n3, bound):  # pylint: disable=C0103
    """Check if 0 <= n1, n2, n3 < bound."""
    return n1 >= 0 and n1 < bound and n2 >= 0 and n2 < bound and n3 >= 0 and n3 < bound


@_numba.njit(cache=True)
def morton(indices):
    """Encode an integer tuple (i1, i2, i3) via Morton encoding."""
    # pylint: disable=C0103
    x, y, z = indices
    return _dilate(x) | (_dilate(y) << 1) | (_dilate(z) << 2)


@_numba.njit(cache=True)
def de_morton(index):
    """Decode a Morton index."""
    ind1 = _contract(index)
    ind2 = _contract(index >> 1)
    ind3 = _contract(index >> 2)

    return (ind1, ind2, ind3)


@_numba.njit(cache=True)
def _neighbors(node_index, level):
    """Return a list of neighbors of a given node."""
    sides = 1 << level

    ind1, ind2, ind3 = de_morton(node_index)

    result = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:
                    continue
                if _in_range(ind1 + i, ind2 + j, ind3 + k, sides):
                    result.append(morton((ind1 + i, ind2 + j, ind3 + k)))
    return result


@_numba.njit(cache=True)
def _dilate(number):
    """Dilate an integer for the Morton encoding."""
    number = (number | (number << 16)) & 0x030000FF
    number = (number | (number << 8)) & 0x0300F00F
    number = (number | (number << 4)) & 0x030C30C3
    number = (number | (number << 2)) & 0x09249249

    # Explanation
    # x = ---- ---- ---- ---- ---- --98 7654 3210
    # x = (x | (x << 16)) & 0x030000FF
    # x = ---- --98 ---- ---- ---- ---- 7654 3210
    # x = (x | (x << 8)) & 0x0300F00F
    # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    # x = (x | (x << 4)) & 0x030C30C3
    # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    # x = (x | (x << 2)) & 0x09249249
    # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

    return number


@_numba.njit(cache=True)
def _contract(number):
    """Undo dilation."""
    number = number & 0x09249249
    number = (number | (number >> 2)) & 0x030C30C3
    number = (number | (number >> 4)) & 0x0300F00F
    number = (number | (number >> 8)) & 0x030000FF
    number = (number | (number >> 16)) & 0x000003FF

    return number

    # Explanation
    #  x &= 0x09249249;
    #  x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    # x = (x | (x >> 2)) & 0x030C30C3
    # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    # x = (x | (x >> 4)) & 0x0300F00F
    # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    # x = (x | (x >> 8)) & 0x030000FF
    # x = ---- --98 ---- ---- ---- ---- 7654 3210
    # x = (x | (x >> 16)) & 0x000003FF
    # x = ---- ---- ---- ---- ---- --98 7654 3210
