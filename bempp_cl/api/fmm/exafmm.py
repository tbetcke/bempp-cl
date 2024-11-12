"""Main interface class to ExaFMM."""

import numpy as _np
import atexit as _atexit

FMM_TMP_DIR = None


@_atexit.register
def cleanup_fmm_tmp():
    """Clean up the FMM tmp directory."""
    from pathlib import Path

    try:
        if FMM_TMP_DIR is not None:
            for tmp_file in Path(FMM_TMP_DIR).glob("*.tmp"):
                tmp_file.unlink()
    except:  # noqa: E722
        print("Could not delete FMM temporary files.")


class ExafmmInterface(object):
    """Interface to Exafmm."""

    def __init__(
        self,
        source_points,
        target_points,
        mode,
        wavenumber=None,
        depth=4,
        expansion_order=5,
        ncrit=400,
        precision="double",
        singular_correction=None,
    ):
        """Instantiate an Exafmm session."""
        import bempp_cl.api
        import os
        from bempp_cl.api.utils.helpers import create_unique_id

        global FMM_TMP_DIR

        if FMM_TMP_DIR is None:
            FMM_TMP_DIR = os.path.join(os.getcwd(), ".exafmm")
            if not os.path.isdir(FMM_TMP_DIR):
                try:
                    os.mkdir(FMM_TMP_DIR)
                except:  # noqa: E722
                    raise FileExistsError(f"A file with the name {FMM_TMP_DIR} exists. Please delete it.")

        for _ in range(10):
            tmp_name = create_unique_id() + ".tmp"
            fname = os.path.join(FMM_TMP_DIR, tmp_name)
            if not os.path.exists(fname):
                break
        else:
            raise FileExistsError("Could not create temporary filename for Exafmm.")

        self._fname = fname
        self._singular_correction = singular_correction

        self._source_points = source_points
        self._target_points = target_points
        self._mode = mode

        if mode == "laplace":
            self._kernel_parameters = _np.array([], dtype="float64")
        elif mode == "helmholtz":
            self._kernel_parameters = _np.array([_np.real(wavenumber), _np.imag(wavenumber)], dtype="float64")
        elif mode == "modified_helmholtz":
            self._kernel_parameters = _np.array([wavenumber], dtype="float64")

        with bempp_cl.api.Timer(message="Initialising Exafmm."):
            if mode == "laplace":
                import exafmm.laplace

                self._module = exafmm.laplace

                sources = exafmm.laplace.init_sources(source_points, _np.zeros(len(source_points), dtype=_np.float64))

                targets = exafmm.laplace.init_targets(target_points)

                self._fmm = exafmm.laplace.LaplaceFmm(expansion_order, ncrit, filename=fname)
                self._tree = exafmm.laplace.setup(sources, targets, self._fmm)

            elif mode == "helmholtz":
                import exafmm.helmholtz

                self._module = exafmm.helmholtz

                sources = exafmm.helmholtz.init_sources(source_points, _np.zeros(len(source_points), dtype=_np.float64))

                targets = exafmm.helmholtz.init_targets(target_points)

                self._fmm = exafmm.helmholtz.HelmholtzFmm(expansion_order, ncrit, wavenumber, filename=fname)
                self._tree = exafmm.helmholtz.setup(sources, targets, self._fmm)

            elif mode == "modified_helmholtz":
                import exafmm.modified_helmholtz

                self._module = exafmm.modified_helmholtz

                sources = exafmm.modified_helmholtz.init_sources(
                    source_points, _np.zeros(len(source_points), dtype=_np.float64)
                )

                targets = exafmm.modified_helmholtz.init_targets(target_points)

                self._fmm = exafmm.modified_helmholtz.ModifiedHelmholtzFmm(
                    expansion_order, ncrit, wavenumber, filename=fname
                )
                self._tree = exafmm.modified_helmholtz.setup(sources, targets, self._fmm)

    @property
    def number_of_source_points(self):
        """Return number of source points."""
        return len(self._source_points)

    @property
    def number_of_target_points(self):
        """Return number of target points."""
        return len(self._target_points)

    def evaluate(self, vec, apply_singular_correction=True):
        """Evalute the Fmm."""
        import bempp_cl.api
        from bempp_cl.api.fmm.helpers import debug_fmm

        with bempp_cl.api.Timer(message="Evaluating Fmm."):
            self._module.update_charges(self._tree, vec)
            self._module.clear_values(self._tree)

            with bempp_cl.api.Timer(message="Calling ExaFMM."):
                if bempp_cl.api.GLOBAL_PARAMETERS.fmm.dense_evaluation:
                    from bempp_cl.api.fmm.helpers import dense_interaction_evaluator

                    result = dense_interaction_evaluator(
                        self._target_points,
                        self._source_points,
                        vec,
                        self._mode,
                        self._kernel_parameters,
                    )
                else:
                    result = self._module.evaluate(self._tree, self._fmm)
                if bempp_cl.api.GLOBAL_PARAMETERS.fmm.debug:
                    debug_fmm(
                        self._target_points,
                        self._source_points,
                        vec,
                        self._mode,
                        self._kernel_parameters,
                        result,
                    )

            if apply_singular_correction and self._singular_correction is not None:
                result -= (self._singular_correction @ vec).reshape([-1, 4])

            return result

    def as_matrix(self):
        """Return matrix representation of Fmm."""
        import numpy as np

        ident = np.identity(self.number_of_source_points)

        res = np.zeros(
            (self.number_of_target_points, self.number_of_source_points),
            dtype="float64",
        )

        for index in range(self.number_of_source_points):
            res[:, index] = self.evaluate(ident[:, index])[:, 0]

        return res

    @classmethod
    def from_grid(
        cls,
        source_grid,
        mode,
        wavenumber=None,
        target_grid=None,
        precision="double",
        parameters=None,
        device_interface=None,
    ):
        """
        Initialise an Exafmm instance from a given source and target grid.

        Parameters
        ----------
        source_grid : Grid object
            Grid for the source points.
        mode: string
            Fmm mode. One of 'laplace', 'helmholtz', or 'modified_helmholtz'
        wavenumber : real number
            For Helmholtz or modified Helmholtz the wavenumber.
        target_grid : Grid object
            An optional target grid. If not provided the source and target
            grid are assumed to be identical.
        precision : string
            Either 'single' or 'double'. Currently, the Fmm is always
            executed in double precision.
        parameters  :   object
            A bempp parameters object. If not provided, the GLOBAL_PARAMETERS will be used.
        device_interface : string
            Either 'numba' or 'opencl'. If not provided, the DEFAULT_DEVICE_INTERFACE
            will be used for the calculation of local interactions.
        """
        import bempp_cl.api
        from bempp_cl.api.integration.triangle_gauss import rule
        from bempp_cl.api.fmm.helpers import get_local_interaction_operator
        import numpy as np

        parameters = bempp_cl.api.assign_parameters(parameters)

        quadrature_order = parameters.quadrature.regular

        local_points, weights = rule(quadrature_order)

        if target_grid is None:
            target_grid = source_grid

        source_points = source_grid.map_to_point_cloud(quadrature_order, precision=precision)

        if target_grid != source_grid:
            target_points = target_grid.map_to_point_cloud(quadrature_order, precision=precision)
        else:
            target_points = source_points

        singular_correction = None

        if target_grid == source_grid:
            # Require singular correction terms.

            if mode == "laplace":
                singular_correction = get_local_interaction_operator(
                    source_grid,
                    local_points,
                    "laplace",
                    np.array([], dtype="float64"),
                    precision,
                    False,
                    device_interface,
                )
            elif mode == "helmholtz":
                singular_correction = get_local_interaction_operator(
                    source_grid,
                    local_points,
                    "helmholtz",
                    np.array([_np.real(wavenumber), _np.imag(wavenumber)], dtype="float64"),
                    precision,
                    True,
                    device_interface,
                )
            elif mode == "modified_helmholtz":
                singular_correction = get_local_interaction_operator(
                    source_grid,
                    local_points,
                    "modified_helmholtz",
                    np.array([wavenumber], dtype="float64"),
                    precision,
                    False,
                    device_interface,
                )
        return cls(
            source_points,
            target_points,
            mode,
            wavenumber=wavenumber,
            depth=parameters.fmm.depth,
            expansion_order=parameters.fmm.expansion_order,
            ncrit=parameters.fmm.ncrit,
            precision=precision,
            singular_correction=singular_correction,
        )
