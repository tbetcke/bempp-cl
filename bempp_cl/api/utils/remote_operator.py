"""Implementation of MPI based remote operators."""

from bempp_cl.api.assembly.blocked_operator import BlockedOperator as _BlockedOperator
from bempp_cl.api.assembly.discrete_boundary_operator import _DiscreteOperatorBase


from mpi4py import MPI
import numpy as _np

MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()

COMM = MPI.COMM_WORLD

_REMOTE_MANAGER = None


class Message:
    """Messages for remote workers."""

    def __init__(self, status, operator_tag=None, is_complex=False, nelements=None):
        """Create message."""
        self.status = status
        self.is_complex = is_complex
        self.operator_tag = operator_tag
        self.nelements = nelements


class RemoteManager:
    """Manage remote worker execution."""

    def __init__(self):
        """Create manager."""
        self._op_data = {}
        self._op_location = {}
        self._tags = {}
        self._tags_counter = 0
        self._rank_counter = 1

    def register(self, op):
        """Register operator for remote execution."""
        tag = self._tags_counter
        rank = self._rank_counter

        self._op_data[tag] = op
        self._op_location[tag] = rank
        self._tags[op] = tag

        self._tags_counter += 1
        self._rank_counter = (1 + self._rank_counter) % MPI_SIZE
        if self._rank_counter == 0:
            self._rank_counter = 1

    def execute_worker(self):
        """Execute on workers."""
        while True:
            msg, tag, data = self.receive_data(0)

            if msg == "SHUTDOWN":
                break
            elif msg == "ASSEMBLE":
                self._op_data[tag].weak_form()
            elif msg == "SYNCHRONIZE":
                self.send_data("SYNCHRONIZED", 0, tag)
            elif msg == "GET_DTYPE":
                self.send_data(_np.dtype(self._op_data[tag].weak_form().dtype).name, 0, tag)
            elif msg == "DATA":
                result = self._op_data[tag].weak_form() @ data
                self.send_data("DATA", 0, tag, result)

    def send_data(self, msg, dest, operator_tag=None, data=None):
        """Send data to destination rank."""
        COMM.send(
            Message(
                msg,
                operator_tag=operator_tag,
                is_complex=_np.iscomplexobj(data),
                nelements=len(data) if data is not None else None,
            ),
            dest=dest,
        )
        if msg == "DATA":
            COMM.Send(data, dest=dest)

    def send_error(self, msg, operator_tag):
        """Send error message to master."""
        COMM.send(Message(msg, operator_tag=operator_tag), dest=0)

    def receive_data(self, source):
        """Receive data."""
        message = COMM.recv(source=source)

        if message.status == "SHUTDOWN":
            return message.status, None, None
        elif message.status == "DATA":
            if message.is_complex:
                dtype = _np.complex128
            else:
                dtype = _np.float64
            data = _np.empty(message.nelements, dtype=dtype)
            COMM.Recv(data, source=source)
            return message.status, message.operator_tag, data
        else:
            return message.status, message.operator_tag, None

    def submit_computation(self, op, x):
        """Submit computation to operator."""
        tag = self._tags[op]
        rank = self._op_location[tag]

        self.send_data("DATA", rank, tag, x)

    def receive_result(self, op):
        """Receive result from a specific operator."""
        tag = self._tags[op]
        rank = self._op_location[tag]
        _, _, data = self.receive_data(rank)
        return data

    def assemble(self, op):
        """Assemble a given remote operator."""
        tag = self._tags[op]
        rank = self._op_location[tag]
        self.send_data("ASSEMBLE", rank, tag)

    def barrier(self):
        """Use barrier operation for workers."""
        for rank in range(1, MPI_SIZE):
            self.send_data("SYNCHRONIZE", rank, 0)
            msg, _, _ = self.receive_data(rank)
            if msg != "SYNCHRONIZED":
                raise Exception(f"Error: expected message 'SYNCHRONIZED' from rank {0}, received" f" {msg}")

    def get_operator_dtype(self, tag):
        """Get dtype of remote operator."""
        rank = self._op_location[tag]
        self.send_data("GET_DTYPE", rank, tag)
        msg, _, _ = self.receive_data(rank)
        return _np.dtype(msg)

    def shutdown(self):
        """Shutdown all workers."""
        if MPI_SIZE == 0:
            return

        for worker in range(1, MPI_SIZE):
            self.send_data("SHUTDOWN", worker)

    def _group_by_rank(self, tags):
        """
        Group tags by for parallel computation.

        Given a list of tags (with possible repetitions).
        Return a list of list of positions in the tags list
        that is sorted so that each sublist does not have repeating
        ranks. This allows to distribute the computations effectively
        on the ranks.
        """
        output = []

        tags_with_indices = {index: tag for index, tag in enumerate(tags)}

        while len(tags_with_indices) > 0:
            current_indices = []
            current_ranks = []
            for index, tag in tags_with_indices.items():
                rank = self._op_location[tag]
                if rank not in current_ranks:
                    current_indices.append(index)
                    current_ranks.append(rank)
            for index in current_indices:
                del tags_with_indices[index]
            output.append(current_indices)

        return output

    def assemble_parallel(self, ops):
        """Assemble a list of operators in parallel."""
        tags = [self._tags[op] for op in ops]
        grouped_indices = self._group_by_rank(tags)

        for index_list in grouped_indices:
            for index in index_list:
                self.assemble(ops[index])
            self.barrier()

    def compute_parallel(self, tasks):
        """
        Compute in parallel.

        Parameters
        ----------
        tasks : list
            A list [(op1, data1), (op2, data2), ...]
            of operators and associated vectors.

        Returns a list [result1, result2, ...]
        of the results of the computation.

        Operators are allowed to appear repeatedly in the list
        and are scheduled correctly.

        """
        tags = [self._tags[op] for op, _ in tasks]

        results = len(tasks) * [None]
        grouped_indices = self._group_by_rank(tags)

        for index_list in grouped_indices:
            for index in index_list:
                op, data = tasks[index]
                self.submit_computation(op, data)
            for index in index_list:
                op, data = tasks[index]
                results[index] = self.receive_result(op)

        return results

    @property
    def tags(self):
        """Return tags."""
        return self._tags


def get_remote_manager():
    """Initialize remote manager."""
    global _REMOTE_MANAGER

    if _REMOTE_MANAGER is None:
        _REMOTE_MANAGER = RemoteManager()
    return _REMOTE_MANAGER


class RemoteBlockedOperator(_BlockedOperator):
    """Define a remote blocked operator."""

    def __init__(self, m, n):
        """Initialize an m x n remote blocked operator."""
        super().__init__(m, n)

    def _assemble(self):
        """Assemble the operator."""
        if not self._fill_complete():
            raise ValueError("Each row and column must have at least one operator")

        return RemoteBlockedDiscreteOperator(self._operators)


class RemoteBlockedDiscreteOperator(_DiscreteOperatorBase):
    """Implementation of a discrete blocked boundary operator."""

    def __init__(self, ops):
        """
        Construct an operator from a two dimensional Numpy array of operators.

        ops is a list of list containing discrete boundary operators or None.
        A None entry is equivalent to a zero discrete boundary operator.

        """
        # pylint: disable=too-many-branches
        self._manager = get_remote_manager()

        if not isinstance(ops, _np.ndarray):
            ops = _np.array(ops)

        rows = ops.shape[0]
        cols = ops.shape[1]

        self._ndims = (rows, cols)

        self._operators = _np.empty((rows, cols), dtype=object)
        self._tags = _np.empty((rows, cols), dtype=_np.int)
        self._rows = -_np.ones(rows, dtype=int)
        self._cols = -_np.ones(cols, dtype=int)

        for i in range(rows):
            for j in range(cols):
                if ops[i, j] is None:
                    continue
                if self._rows[i] != -1:
                    if self._get_shape(ops[i, j])[0] != self._rows[i]:
                        raise ValueError("Block row {0} has incompatible ".format(i) + " operator sizes.")
                else:
                    self._rows[i] = self._get_shape(ops[i, j])[0]

                if self._cols[j] != -1:
                    if self._get_shape(ops[i, j])[1] != self._cols[j]:
                        raise ValueError("Block column {0} has incompatible".format(j) + "operator sizes.")
                else:
                    self._cols[j] = self._get_shape(ops[i, j])[1]
                self._operators[i, j] = ops[i, j]

        if not self._fill_complete():
            raise ValueError("Each row and column must contain at least one operator.")

        for i in range(rows):
            for j in range(cols):
                if self._operators[i, j] is None:
                    self._tags[i, j] = -1
                else:
                    self._tags[i, j] = self._manager.tags[self._operators[i, j]]

        shape = (_np.sum(self._rows), _np.sum(self._cols))

        is_complex = False
        for i in range(rows):
            for j in range(cols):
                if self._tags[i, j] == -1:
                    continue
                self._manager.assemble(self._operators[i, j])

        self._manager.barrier()

        for i in range(rows):
            for j in range(cols):
                if self._tags[i, j] == -1:
                    continue
                dtype = self._manager.get_operator_dtype(self._tags[i, j])
                if dtype in ["complex128", "complex64"]:
                    is_complex = True

        if is_complex:
            dtype = _np.complex128
        else:
            dtype = _np.float64

        super().__init__(dtype, shape)

    def __getitem__(self, key):
        """Return the object at position (i, j)."""
        return self._operators[key]

    def _get_shape(self, op):
        """Get shape of boundary operator."""
        return (op.dual_to_range.global_dof_count, op.domain.global_dof_count)

    def _fill_complete(self):
        """Check if the fill is complete."""
        if (-1 in self._rows) or (-1 in self._cols):
            return False
        return True

    def _matvec(self, x):
        """Perform matrix vector multiplication."""
        from bempp_cl.api.utils.data_types import combined_type

        if not self._fill_complete():
            raise ValueError("Not all rows or columns contain operators.")

        ndims = len(x.shape)
        x = x.ravel()

        row_dim = 0
        res = _np.zeros(self.shape[0], dtype=combined_type(self.dtype, x.dtype))

        compute_tasks = []

        # Submit the computations
        for i in range(self._ndims[0]):
            col_dim = 0
            for j in range(self._ndims[1]):
                if self._tags[i, j] != -1:
                    # If self._tags[i, j] == -1 the operator is None
                    local_x = x[col_dim : col_dim + self._cols[j]]
                    compute_tasks.append((self._operators[i, j], local_x))
                col_dim += self._cols[j]
            row_dim += self._rows[i]

        # Execute computations

        compute_results = self._manager.compute_parallel(compute_tasks)

        # Get results back
        row_dim = 0
        task_count = 0
        for i in range(self._ndims[0]):
            col_dim = 0
            for j in range(self._ndims[1]):
                if self._tags[i, j] == -1:
                    continue
                res[row_dim : row_dim + self._rows[i]] += compute_results[task_count]
                task_count += 1
            row_dim += self._rows[i]

        if ndims == 2:
            res = res.reshape(-1, 1)

        return res

    def _get_row_dimensions(self):
        """Return the number of rows."""
        return self._rows

    def _get_column_dimensions(self):
        """Return the number of columns."""
        return self._cols

    @property
    def A(self):
        """Return as dense Numpy array."""
        raise NotImplementedError()

    def _transpose(self):
        """Implement the transpose."""
        raise NotImplementedError()

    def _adjoint(self):
        """Implement the adjoint."""
        raise NotImplementedError()

    row_dimensions = property(_get_row_dimensions)
    column_dimensions = property(_get_column_dimensions)
