"""Define interfaces to external viewers."""

import numpy as _np


def visualize(obj, mode=None, transformation=None):
    """
    Create a visualisation.

    Attributes
    ----------
    obj : Grid or GridFunction object
        A grid or grid function to plot
    mode : string
        One of 'element' or 'node'. If 'element' is chosen
        the color is determined by the mid-point of the faces
        of the grid. For 'node' the vertex values are
        chosen. Only used for grid functions.
    transformation : string or object
        One of 'real', 'imag', 'abs', 'log_abs' or
        'abs_squared' or a callable object.
        Describes the data transformation
        before plotting. For functions with vector values
        only 'abs', 'log_abs' or 'abs_squared' are allowed.
        If a callable object is given this is applied instead.
        It is important that the callable returns numpy arrays
        with the same number of dimensions as before.

    """
    import bempp_cl.api
    import numpy as np

    transform = None

    if transformation == "real":
        transform = np.real
    elif transformation == "imag":
        transform = np.imag
    elif transformation == "abs":
        transform = lambda x: np.sqrt(np.sum(np.abs(x) ** 2, axis=0, keepdims=True))
    elif transformation == "log_abs":
        transform = lambda x: np.log(np.sqrt(np.sum(np.abs(x) ** 2, axis=0, keepdims=True)))
    elif transformation == "abs_squared":
        transform = lambda x: np.sum(np.abs(x) ** 2, axis=0, keepdims=True)
    else:
        transform = transformation

    if bempp_cl.api.PLOT_BACKEND == "gmsh":
        visualize_with_gmsh(obj, mode, transform)
    if bempp_cl.api.PLOT_BACKEND == "paraview":
        visualize_with_paraview(obj, mode, transform)
    if bempp_cl.api.PLOT_BACKEND == "jupyter_notebook":
        visualize_with_jupyter_notebook(obj, mode, transform)


def visualize_with_jupyter_notebook(obj, mode=None, transformation=None):
    """View a grid or grid function in an IPython Notebook."""
    import plotly
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    from bempp_cl.api import GridFunction
    from bempp_cl.api.grid.grid import Grid
    import numpy as np

    if transformation is None:
        transformation = np.real

    plotly.offline.init_notebook_mode()

    if isinstance(obj, Grid):
        vertices = obj.vertices
        elements = obj.elements
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=elements.shape[1] * ["rgb(255, 222, 173)"],
        )
        fig["layout"]["scene"].update(go.layout.Scene(aspectmode="data"))
        plotly.offline.iplot(fig)

    elif isinstance(obj, GridFunction):
        import matplotlib as mpl
        from matplotlib import pyplot as plt

        cmap = plt.get_cmap("jet")

        grid = obj.space.grid
        vertices = grid.vertices
        elements = grid.elements

        local_coordinates = _np.array([[1.0 / 3], [1.0 / 3]])
        values = _np.zeros(grid.entity_count(0), dtype="float64")
        for element in grid.entity_iterator(0):
            index = element.index
            local_values = np.real(transformation(obj.evaluate(index, local_coordinates)))
            values[index] = local_values.flatten()

        norm = mpl.colors.Normalize(vmin=_np.min(values), vmax=_np.max(values))
        colorfun = lambda x: _np.rint(_np.array(cmap(norm(x))) * 255)
        color_codes = ["rgb({0}, {1}, {2})".format(*colorfun(x)) for x in values]
        fig = ff.create_trisurf(
            x=vertices[0, :],
            y=vertices[1, :],
            z=vertices[2, :],
            simplices=elements.T,
            color_func=color_codes,
        )
        fig["layout"]["scene"].update(go.layout.Scene(aspectmode="data"))
        plotly.offline.iplot(fig)


def visualize_with_gmsh(obj, mode=None, transformation=None):
    """
    View a grid or grid function with Gmsh.

    Parameters
    ----------
    obj : bempp_cl.api.Grid or bempp_cl.api.GridFunction
        Grid or grid function to visualize.
    mode : string
        One of 'element' or 'node'
        (default 'vertices')
    transformation : callable
        A function object that is applied to the data before
        writing it out

    Notes
    -----
    This function writes the data into a temp file and
    visualizes it.

    """
    import tempfile
    import subprocess
    from bempp_cl.api import export, GMSH_PATH, TMP_PATH, GridFunction
    from bempp_cl.api.grid.grid import Grid

    if GMSH_PATH is None:
        print("Gmsh not available for visualization.")
        return None

    outfile = tempfile.NamedTemporaryFile(suffix=".msh", dir=TMP_PATH, delete=False)
    if isinstance(obj, Grid):
        export(outfile.name, grid=obj)
    elif isinstance(obj, GridFunction):
        export(
            outfile.name,
            grid_function=obj,
            transformation=transformation,
            data_type=mode,
        )
    outfile.close()

    subprocess.Popen([GMSH_PATH, outfile.name])


def visualize_with_paraview(obj, mode=None, transformation=None):
    """
    View a grid or grid function with Paraview.

    Parameters
    ----------
    obj : bempp_cl.api.Grid or bempp_cl.api.GridFunction
        Grid or grid function to visualize.
    mode : string
        One of 'element' or 'node'
        (default 'vertices')
    transformation : callable
        A function object that is applied to the data before
        writing it out

    Notes
    -----
    This function writes the data into a temp file and
    visualizes it.

    """
    import tempfile
    import subprocess
    from bempp_cl.api import export, TMP_PATH, GridFunction
    from bempp_cl.api.grid.grid import Grid
    from bempp_cl.api.utils import which
    import os

    if os.name == "nt":
        pview = which("paraview.exe")
    else:
        pview = which("paraview")

    if pview is None:
        raise EnvironmentError("Could not find Paraview." + "Interactive plotting with Paraview not available.")

    outfile = tempfile.NamedTemporaryFile(suffix=".vtu", dir=TMP_PATH, delete=False)
    if isinstance(obj, Grid):
        export(outfile.name, grid=obj)
    elif isinstance(obj, GridFunction):
        export(
            outfile.name,
            grid_function=obj,
            transformation=transformation,
            data_type=mode,
        )
    outfile.close()

    subprocess.Popen([pview, outfile.name])


def enable_gmsh_viewer():
    """Change plotting default to Gmsh."""
    import bempp_cl.api

    bempp_cl.api.PLOT_BACKEND = "gmsh"


def enable_paraview_viewer():
    """Change plotting default to Paraview."""
    import bempp_cl.api

    bempp_cl.api.PLOT_BACKEND = "paraview"


def enable_jupyter_viewer():
    """Change plotting default to IPython."""
    import bempp_cl.api

    bempp_cl.api.PLOT_BACKEND = "jupyter_notebook"
