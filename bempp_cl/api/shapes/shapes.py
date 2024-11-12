"""Various built-in test shapes."""

import numpy as _np


def get_gmsh_file():
    """
    Create a new temporary gmsh file.

    Return a 3-tuple (geo_file,geo_name,msh_name), where
    geo_file is a file descriptor to an empty .geo file, geo_name is
    the corresponding filename and msh_name is the name of the
    Gmsh .msh file that will be generated.

    """
    import os
    import tempfile
    import bempp_cl.api

    geo, geo_name = tempfile.mkstemp(suffix=".geo", dir=bempp_cl.api.TMP_PATH, text=True)
    geo_file = os.fdopen(geo, "w")
    msh_name = os.path.splitext(geo_name)[0] + ".msh"
    return (geo_file, geo_name, msh_name)


def __generate_grid_from_gmsh_string(gmsh_string):
    """Return a grid from a string containing a gmsh mesh."""
    import os
    import tempfile
    import bempp_cl.api

    if bempp_cl.api.mpi_rank == 0:
        # First create the grid.
        handle, fname = tempfile.mkstemp(suffix=".msh", dir=bempp_cl.api.TMP_PATH, text=True)
        with os.fdopen(handle, "w") as f:
            f.write(gmsh_string)
    grid = bempp_cl.api.import_grid(fname)
    bempp_cl.api.mpi_comm.Barrier()
    if bempp_cl.api.mpi_rank == 0:
        os.remove(fname)
    return grid


def __generate_grid_from_geo_string(geo_string):
    """Create a grid from a gmsh geo string."""
    import os
    import subprocess
    import bempp_cl.api

    def msh_from_string(geo_string):
        """Create a mesh from a string."""
        gmsh_command = bempp_cl.api.GMSH_PATH
        if gmsh_command is None:
            raise RuntimeError("Gmsh is not found. Cannot generate mesh")
        f, geo_name, msh_name = get_gmsh_file()
        f.write(geo_string)
        f.close()

        fnull = open(os.devnull, "w")
        cmd = gmsh_command + " -2 " + geo_name
        try:
            subprocess.check_call(cmd, shell=True, stdout=fnull, stderr=fnull)
        except:  # noqa: E722
            print("The following command failed: " + cmd)
            fnull.close()
            raise
        os.remove(geo_name)
        fnull.close()
        return msh_name

    msh_name = msh_from_string(geo_string)
    grid = bempp_cl.api.import_grid(msh_name)
    os.remove(msh_name)
    return grid


def screen(corners, h=0.1):
    """
    Create a screen.

    Parameters
    ----------
    corners : np.ndarray
        A (4 x 3) array that defines four corners of the screen.
    h : float
        A floating point number specifying the grid size.

    Output
    -------
    grid : bempp_cl.Grid
        A structured grid.

    """
    stub = f"""
    cl = {h};
    Point(1) = {{ {corners[0, 0]}, {corners[0, 1]}, {corners[0, 2]}, cl }};
    Point(2) = {{ {corners[1, 0]}, {corners[1, 1]}, {corners[1, 2]}, cl }};
    Point(3) = {{ {corners[2, 0]}, {corners[2, 1]}, {corners[2, 2]}, cl }};
    Point(4) = {{ {corners[3, 0]}, {corners[3, 1]}, {corners[3, 2]}, cl }};

    Line(1) = {{1,2}};
    Line(2) = {{2,3}};
    Line(3) = {{3,4}};
    Line(4) = {{4,1}};
    Line Loop(1) = {{1, 2, 3, 4}};
    Plane Surface(1) = {{1}};
    Physical Surface(1) = {{1}};

    Mesh.Algorithm = 6;
    """
    return __generate_grid_from_geo_string(stub)


def regular_sphere(refine_level):
    """
    Create a regular sphere with a given refinement level.

    Starting from an octahedron with 8 elements the grid is
    refined in each step by subdividing each element into
    four new elements, to create a sphere approximation.

    The number of elements in the final sphere is given as
    8 * 4**refine_level.

    The maximum allowed refinement level is 9.

    """
    from bempp_cl.api.grid.grid import Grid
    import os
    import numpy as np

    if refine_level > 9:
        raise ValueError("'refine_level larger than 9 not supported.")

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "regular_spheres.npz")

    spheres = np.load(filename)
    return Grid(spheres["v" + str(refine_level)], spheres["e" + str(refine_level)])


def multitrace_cube(h=0.1):
    """
    Definitition of a cube with an interface at z=.5.

    The normal direction at the interface shows into the
    positive z-direction and has the domain index
    and has the domain index 11. The lower half of the cube
    is given through the segments [1, 2, 3, 4, 5, 6]. The
    top half of the cube is defined by the segments
    [6, 7, 8, 9, 10, 11]. For the upper half the normal
    direction of segment 6 shows in the interior of the domain.
    """
    stub = """
    Point(1) = {0, 0.0, 0, cl};
    Point(2) = {1, 0, 0, cl};
    Point(3) = {1, 1, 0, cl};
    Point(4) = {0, 1, 0, cl};
    Point(5) = {1, 0, 1, cl};
    Point(6) = {0, 1, 1, cl};
    Point(7) = {1, 1, 1, cl};
    Point(8) = {0, 0, 1, cl};
    Point(9) = {1, 0, .5, cl};
    Point(10) = {0, 1, .5, cl};
    Point(11) = {1, 1, .5, cl};
    Point(12) = {0, 0, .5, cl};
    Line(1) = {8, 5};
    Line(3) = {2, 1};
    Line(5) = {6, 7};
    Line(7) = {3, 4};
    Line(9) = {7, 5};
    Line(10) = {6, 8};
    Line(11) = {3, 2};
    Line(12) = {4, 1};
    Line(13) = {12, 9};
    Line(14) = {9, 11};
    Line(15) = {11, 10};
    Line(16) = {10, 12};
    Line(17) = {2, 9};
    Line(18) = {3, 11};
    Line(19) = {11, 7};
    Line(20) = {9, 5};
    Line(21) = {4, 10};
    Line(22) = {1, 12};
    Line(23) = {12, 8};
    Line(24) = {10, 6};
    Line Loop(1) = {3, -12, -7, 11};
    Plane Surface(1) = {1};
    Line Loop(3) = {14, 19, 9, -20};
    Plane Surface(3) = {3};
    Line Loop(4) = {13, 20, -1, -23};
    Plane Surface(4) = {4};
    Line Loop(6) = {12, 22, -16, -21};
    Plane Surface(6) = {6};
    Line Loop(7) = {16, 23, -10, -24};
    Plane Surface(7) = {7};
    Line Loop(9) = {7, 21, -15, -18};
    Plane Surface(9) = {9};
    Line Loop(10) = {15, 24, 5, -19};
    Plane Surface(10) = {10};
    Line Loop(11) = {16, 13, 14, 15};
    Plane Surface(11) = {11};
    Line Loop(12) = {1, -9, -5, 10};
    Plane Surface(12) = {12};
    Line Loop(13) = {-3, 17, -13, -22};
    Plane Surface(13) = {13};
    Line Loop(14) = {-11, 18, -14, -17};
    Plane Surface(14) = {14};
    Physical Surface(1) = {6};
    Physical Surface(2) = {13};
    Physical Surface(3) = {14};
    Physical Surface(4) = {9};
    Physical Surface(5) = {1};
    Physical Surface(6) = {11};
    Physical Surface(7) = {3};
    Physical Surface(8) = {10};
    Physical Surface(9) = {7};
    Physical Surface(10) = {4};
    Physical Surface(11) = {12};
    """
    geometry = "cl = " + str(h) + ";\n" + stub
    return __generate_grid_from_geo_string(geometry)


def reference_triangle():
    """Return a grid consisting of only the reference triangle."""
    from bempp_cl.api.grid.grid import Grid

    vertices = _np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).T

    elements = _np.array([[0, 1, 2]]).T

    return Grid(vertices, elements)


def ellipsoid(r1=1, r2=1, r3=1, origin=(0, 0, 0), h=0.1):
    """
    Return an ellipsoid grid.

    Parameters
    ----------
    r1 : float
        Radius of first major axis
    r2 : float
        Radius of second major axis
    r3 : float
        Radius of third major axis
    origin : tuple
        Tuple specifying the origin of the ellipsoid
    h : float
        Element size.
    """
    stub = """
    Point(1) = {orig0,orig1,orig2,cl};
    Point(2) = {orig0+r1,orig1,orig2,cl};
    Point(3) = {orig0,orig1+r2,orig2,cl};
    Ellipse(1) = {2,1,2,3};
    Point(4) = {orig0-r1,orig1,orig2,cl};
    Point(5) = {orig0,orig1-r2,orig2,cl};
    Ellipse(2) = {3,1,4,4};
    Ellipse(3) = {4,1,4,5};
    Ellipse(4) = {5,1,2,2};
    Point(6) = {orig0,orig1,orig2-r3,cl};
    Point(7) = {orig0,orig1,orig2+r3,cl};
    Ellipse(5) = {3,1,3,6};
    Ellipse(6) = {6,1,5,5};
    Ellipse(7) = {5,1,5,7};
    Ellipse(8) = {7,1,3,3};
    Ellipse(9) = {2,1,2,7};
    Ellipse(10) = {7,1,4,4};
    Ellipse(11) = {4,1,4,6};
    Ellipse(12) = {6,1,2,2};
    Line Loop(13) = {2,8,-10};
    Ruled Surface(14) = {13};
    Line Loop(15) = {10,3,7};
    Ruled Surface(16) = {15};
    Line Loop(17) = {-8,-9,1};
    Ruled Surface(18) = {17};
    Line Loop(19) = {-11,-2,5};
    Ruled Surface(20) = {19};
    Line Loop(21) = {-5,-12,-1};
    Ruled Surface(22) = {21};
    Line Loop(23) = {-3,11,6};
    Ruled Surface(24) = {23};
    Line Loop(25) = {-7,4,9};
    Ruled Surface(26) = {25};
    Line Loop(27) = {-4,12,-6};
    Ruled Surface(28) = {27};
    Surface Loop(29) = {28,26,16,14,20,24,22,18};
    Volume(30) = {29};
    Physical Surface(10) = {28,26,16,14,20,24,22,18};
    Mesh.Algorithm = 6;
    """

    geometry = (
        "r1 = "
        + str(r1)
        + ";\n"
        + "r2 = "
        + str(r2)
        + ";\n"
        + "r3 = "
        + str(r3)
        + ";\n"
        + "orig0 = "
        + str(origin[0])
        + ";\n"
        + "orig1 = "
        + str(origin[1])
        + ";\n"
        + "orig2 = "
        + str(origin[2])
        + ";\n"
        + "cl = "
        + str(h)
        + ";\n"
        + stub
    )

    return __generate_grid_from_geo_string(geometry)


def multitrace_ellipsoid(r1=1, r2=1, r3=1, origin=(0, 0, 0), h=0.1):
    """
    Return an ellipsoid grid.

    Parameters
    ----------
    r1 : float
        Radius of first major axis
    r2 : float
        Radius of second major axis
    r3 : float
        Radius of third major axis
    origin : tuple
        Tuple specifying the origin of the ellipsoid
    h : float
        Element size.
    """
    stub = """
    Point(1) = {orig0,orig1,orig2,cl};
    Point(2) = {orig0+r1,orig1,orig2,cl};
    Point(3) = {orig0,orig1+r2,orig2,cl};
    Ellipse(1) = {2,1,2,3};
    Point(4) = {orig0-r1,orig1,orig2,cl};
    Point(5) = {orig0,orig1-r2,orig2,cl};
    Ellipse(2) = {3,1,4,4};
    Ellipse(3) = {4,1,4,5};
    Ellipse(4) = {5,1,2,2};
    Point(6) = {orig0,orig1,orig2-r3,cl};
    Point(7) = {orig0,orig1,orig2+r3,cl};
    Ellipse(5) = {3,1,3,6};
    Ellipse(6) = {6,1,5,5};
    Ellipse(7) = {5,1,5,7};
    Ellipse(8) = {7,1,3,3};
    Ellipse(9) = {2,1,2,7};
    Ellipse(10) = {7,1,4,4};
    Ellipse(11) = {4,1,4,6};
    Ellipse(12) = {6,1,2,2};
    Line Loop(13) = {2,8,-10};
    Ruled Surface(14) = {13};
    Line Loop(100) = {5,6,7,8};
    Ruled Surface(101) = {-100};
    Line Loop(15) = {10,3,7};
    Ruled Surface(16) = {15};
    Line Loop(17) = {-8,-9,1};
    Ruled Surface(18) = {17};
    Line Loop(19) = {-11,-2,5};
    Ruled Surface(20) = {19};
    Line Loop(21) = {-5,-12,-1};
    Ruled Surface(22) = {21};
    Line Loop(23) = {-3,11,6};
    Ruled Surface(24) = {23};
    Line Loop(25) = {-7,4,9};
    Ruled Surface(26) = {25};
    Line Loop(27) = {-4,12,-6};
    Ruled Surface(28) = {27};
    Surface Loop(29) = {28,26,16,14,20,24,22,18};
    Volume(30) = {29};
    Physical Surface(10) = {14,16,24,20};
    Physical Surface(20) = {18,22,26,28};
    Physical Surface(12) = {101};
    Mesh.Algorithm = 6;
    """

    geometry = (
        "r1 = "
        + str(r1)
        + ";\n"
        + "r2 = "
        + str(r2)
        + ";\n"
        + "r3 = "
        + str(r3)
        + ";\n"
        + "orig0 = "
        + str(origin[0])
        + ";\n"
        + "orig1 = "
        + str(origin[1])
        + ";\n"
        + "orig2 = "
        + str(origin[2])
        + ";\n"
        + "cl = "
        + str(h)
        + ";\n"
        + stub
    )

    return __generate_grid_from_geo_string(geometry)


def sphere(r=1, origin=(0, 0, 0), h=0.1):
    """
    Return a sphere grid.

    Parameters
    ----------
    r : float
        Radius of the sphere.
    origin : tuple
        Center of the sphere.
    h : float
        Element size.

    """
    return ellipsoid(r1=r, r2=r, r3=r, origin=origin, h=h)


def multitrace_sphere(r=1, origin=(0, 0, 0), h=0.1):
    """
    Return a multitrace sphere grid.

    Parameters
    ----------
    r : float
        Radius of the sphere.
    origin : tuple
        Center of the sphere.
    h : float
        Element size.

    """
    return multitrace_ellipsoid(r1=r, r2=r, r3=r, origin=origin, h=h)


def rectangle_with_hole(a=1, b=1, hole_radius=0.2, h=0.1):
    """
    Return a square shaped screen with a hole in the middle.

    a : float
        Length of rectangle in the x-plane.
    b : float
        Length of rectange in the y-plane.
    hole_radius : float
        Radius of the hole.
    h : float
        Element size.

    """
    stub = """
    Point(1) = {-a / 2., -b / 2., 0, cl};
    Point(2) = {a / 2., -b / 2., 0, cl};
    Point(3) = {a / 2., b / 2., 0, cl};
    Point(4) = {-a / 2., b / 2., 0, cl};
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};
    Point(5) = {0, 0, 0, cl};
    Point(6) = {r, 0, 0, cl};
    Point(7) = {0, r, 0, cl};
    Point(8) = {-r, 0, 0, cl};
    Point(9) = {0, -r, 0, cl};
    Circle(5) = {6, 5, 7};
    Circle(6) = {7, 5, 8};
    Circle(7) = {8, 5, 9};
    Circle(8) = {9, 5, 6};
    Line Loop(9) = {1, 2, 3, 4};
    Line Loop(10) = {5, 6, 7, 8};
    Plane Surface(11) = {9, 10};
    Mesh.Algorithm = 6;
    """

    geometry = (
        "a = "
        + str(a)
        + ";\n"
        + "b = "
        + str(b)
        + ";\n"
        + "r = "
        + str(hole_radius)
        + ";\n"
        + "cl = "
        + str(h)
        + ";\n"
        + stub
    )

    return __generate_grid_from_geo_string(geometry)


def reentrant_cube(h=0.1, refinement_factor=0.2):
    """
    Create a reentrant corner in 3d.

    Parameters
    ----------
    h : float
        Element size.
    refinement_factor : float
        Fractional size with respect to h of elements close to reentrant
        corner.
    """
    reentrant_cube_stub = """
    Point(1) = {0, 0, 0, h};
    Point(2) = {1, 0, 0, h};
    Point(3) = {1, 1, 0, h};
    Point(4) = {0, 1, 0, h};
    Point(5) = {0, 0, 1, h};
    Point(6) = {1, 0, 1, h};
    Point(7) = {1, 1, 1, h};
    Point(8) = {0, 1, 1, h};
    Point(9) = {.5, .5, .5, r};
    Point(10) = {0, 0, .5, h};
    Point(11) = {0.5, 0, .5, h};
    Point(12) = {0, .5, .5, h};
    Point(13) = {0.5, 0, 1, h};
    Point(14) = {0.5, 0.5, 1, h};
    Point(15) = {0, 0.5, 1, h};

    Line(1) = {1, 2};
    Line(2) = {2, 6};
    Line(3) = {6, 13};
    Line(4) = {13, 11};
    Line(5) = {11, 10};
    Line(6) = {10, 1};

    Line(7) = {2, 3};
    Line(8) = {3, 7};
    Line(9) = {7, 6};

    Line(10) = {3, 4};
    Line(11) = {4, 8};
    Line(12) = {8, 7};

    Line(13) = {4, 1};
    Line(14) = {10, 12};
    Line(15) = {12, 15};
    Line(16) = {15, 8};

    Line(17) = {12, 9};
    Line(18) = {9, 14};
    Line(19) = {14, 15};

    Line(20) = {11, 9};
    Line(21) = {13, 14};

    Line Loop(1) = {1, 2, 3, 4, 5, 6};
    Line Loop(2) = {-2, 7, 8, 9};
    Line Loop(3) = {-8, 10, 11, 12};
    Line Loop(4) = {13, -6, 14, 15, 16, -11};
    Line Loop(5) = {17, 18, 19, -15};
    Line Loop(6) = {-14, -5, 20, -17};
    Line Loop(7) = {-4, 21, -18, -20};
    Line Loop(8) = {-13, -10, -7, -1};
    Line Loop(9) = {-21, -3, -9, -12, -16, -19};

    Plane Surface(1) = {1};
    Plane Surface(2) = {2};
    Plane Surface(3) = {3};
    Plane Surface(4) = {4};
    Plane Surface(5) = {5};
    Plane Surface(6) = {6};
    Plane Surface(7) = {7};
    Plane Surface(8) = {8};
    Plane Surface(9) = {9};

    Surface Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    Volume(1) = {1};
    Mesh.Algorithm = 6;
    """
    reentrant_cube_geometry = (
        "h = " + str(h) + ";\n" + "r = h * " + str(refinement_factor) + ";\n" + reentrant_cube_stub
    )
    return __generate_grid_from_geo_string(reentrant_cube_geometry)


def cuboid(length=(1, 1, 1), origin=(0, 0, 0), h=0.1):
    """
    Return a cuboid mesh.

    Parameters
    ----------
    length : tuple
        Side lengths of the cube.
    origin : tuple
        Coordinates of the origin (bottom left corner)
    h : float
        Element size.

    """
    cuboid_stub = """
    Point(1) = {orig0,orig1,orig2,cl};
    Point(2) = {orig0+l0,orig1,orig2,cl};
    Point(3) = {orig0+l0,orig1+l1,orig2,cl};
    Point(4) = {orig0,orig1+l1,orig2,cl};
    Point(5) = {orig0,orig1,orig2+l2,cl};
    Point(6) = {orig0+l0,orig1,orig2+l2,cl};
    Point(7) = {orig0+l0,orig1+l1,orig2+l2,cl};
    Point(8) = {orig0,orig1+l1,orig2+l2,cl};

    Line(1) = {1,2};
    Line(2) = {2,3};
    Line(3) = {3,4};
    Line(4) = {4,1};
    Line(5) = {1,5};
    Line(6) = {2,6};
    Line(7) = {3,7};
    Line(8) = {4,8};
    Line(9) = {5,6};
    Line(10) = {6,7};
    Line(11) = {7,8};
    Line(12) = {8,5};

    Line Loop(1) = {-1,-4,-3,-2};
    Line Loop(2) = {1,6,-9,-5};
    Line Loop(3) = {2,7,-10,-6};
    Line Loop(4) = {3,8,-11,-7};
    Line Loop(5) = {4,5,-12,-8};
    Line Loop(6) = {9,10,11,12};

    Plane Surface(1) = {1};
    Plane Surface(2) = {2};
    Plane Surface(3) = {3};
    Plane Surface(4) = {4};
    Plane Surface(5) = {5};
    Plane Surface(6) = {6};

    Physical Surface(1) = {1};
    Physical Surface(2) = {2};
    Physical Surface(3) = {3};
    Physical Surface(4) = {4};
    Physical Surface(5) = {5};
    Physical Surface(6) = {6};

    Surface Loop (1) = {1,2,3,4,5,6};

    Volume (1) = {1};

    Mesh.Algorithm = 6;
    """

    cuboid_geometry = (
        "l0 = "
        + str(length[0])
        + ";\n"
        + "l1 = "
        + str(length[1])
        + ";\n"
        + "l2 = "
        + str(length[2])
        + ";\n"
        + "orig0 = "
        + str(origin[0])
        + ";\n"
        + "orig1 = "
        + str(origin[1])
        + ";\n"
        + "orig2 = "
        + str(origin[2])
        + ";\n"
        + "cl = "
        + str(h)
        + ";\n"
        + cuboid_stub
    )

    return __generate_grid_from_geo_string(cuboid_geometry)


def cube(length=1, origin=(0, 0, 0), h=0.1):
    """
    Return a cube mesh.

    Parameters
    ----------
    length : float
        Side length of the cube.
    origin : tuple
        Coordinates of the origin (bottom left corner)
    h : float
        Element size.

    """
    return cuboid((length, length, length), origin, h)


def almond(h=0.01):
    """Return the Nasa almond shape with element size h."""
    almond_geometry = "cl = {0};\n".format(h) + _almond_geo
    return __generate_grid_from_geo_string(almond_geometry)


_almond_geo = """
Function QuarterEllipse1
x = d*t*t_fact;
y = 4.83345 * d * ( Sqrt(1-(t * t_fact/2.08335)^2)-0.96);
z = 1.61115 * d * ( Sqrt(1-(t * t_fact/2.08335)^2)-0.96);

Point(point_number) = {x, 0, 0, cl};
CenterNumber = point_number;
Psi_array[] = {0.0, 15.0, 45.0, 90.0};

For i In {0:3}
point_number = newp;
thePointNumber[i] = point_number;
psi = Psi_array[i]/180.0*Pi;
Point(point_number) = {x, y*Cos(psi), z*Sin(psi), cl};
point_number = newp;
EndFor

For i In {0:2}
Ellipse(newreg) = {thePointNumber[i],CenterNumber,thePointNumber[3],thePointNumber[i+1]};
EndFor
Return

Function QuarterEllipse2
x = d*t * t_fact;
y = yfact * d * (Sqrt(1-(t*t_fact*3.0/1.25)^2));
z = zfact * d * (Sqrt(1-(t*t_fact*3.0/1.25)^2));

Point(point_number) = {x, 0, 0, cl};
CenterNumber = point_number;
Psi_array[] = {0.0, 15.0, 45.0, 90.0};

For i In {0:3}
point_number = newp;
thePointNumber[i] = point_number;
psi = Psi_array[i]/180.0*Pi;
Point(point_number) = {x, y*Cos(psi), z*Sin(psi), cl};
point_number = newp;
EndFor

For i In {0:2}
Ellipse(newreg) = {thePointNumber[i],CenterNumber,thePointNumber[3],thePointNumber[i+1]};
EndFor
Return

d = 9.936 * 0.0254;

t = 1.75/3.0;

Point(1) = {d*t, 0.0, 0.0, cl};
point_number = 2;

lc_fact = 0.1;
t_fact = 0.975;
Call QuarterEllipse1;

lc_fact = 0.75;
t_fact = 0.95;
Call QuarterEllipse1;

lc_fact = 1.0;
t_fact = 0.9;
For j In {1:10}
   Call QuarterEllipse1;
   t_fact -= 0.1;
EndFor

t = -1.25/3.0;
yfact = 0.58/3.0;
zfact = 0.58/9.0;

lc_fact = 1.0;
t_fact = 0.1;

For j In {1:9}
    Call QuarterEllipse2;
        t_fact += 0.1;
EndFor

lc_fact = 0.1;
t_fact = 0.95;
Call QuarterEllipse2;

lc_fact = 0.75;
t_fact = 0.99;
Call QuarterEllipse2;


Point(point_number) = {d*t, 0.0, 0.0, cl};
Spline(70) = {117, 115, 110};
Spline(71) = {111, 116, 117};
Spline(74) = {117, 113, 108};
Spline(75) = {117, 114, 109};
Spline(76) = {108, 103, 98};
Spline(77) = {109, 104, 99};
Spline(78) = {110, 105, 100};
Spline(79) = {111, 106, 101};
Spline(80) = {101, 96, 91};
Spline(81) = {91, 86, 81};
Spline(82) = {81, 76, 71};
Spline(83) = {71, 66, 61};
Spline(84) = {61, 56, 51};
Spline(85) = {51, 46, 41};
Spline(86) = {41, 36, 31};
Spline(87) = {31, 26, 21};
Spline(88) = {21, 16, 11};
Spline(89) = {11, 6, 1};
Spline(90) = {1, 5, 10};
Spline(91) = {10, 15, 20};
Spline(92) = {20, 25, 30};
Spline(93) = {30, 35, 40};
Spline(94) = {40, 45, 50};
Spline(95) = {50, 55, 60};
Spline(96) = {60, 65, 70};
Spline(97) = {70, 75, 80};
Spline(98) = {80, 85, 90};
Spline(99) = {90, 95, 100};
Spline(100) = {99, 94, 89};
Spline(101) = {89, 84, 79};
Spline(102) = {79, 74, 69};
Spline(103) = {69, 64, 59};
Spline(104) = {59, 54, 49};
Spline(105) = {49, 44, 39};
Spline(106) = {39, 34, 29};
Spline(107) = {29, 24, 19};
Spline(108) = {19, 14, 9};
Spline(109) = {9, 4, 1};
Spline(110) = {1, 3, 8};
Spline(111) = {8, 13, 18};
Spline(112) = {18, 23, 28};
Spline(113) = {28, 33, 38};
Spline(114) = {38, 43, 48};
Spline(115) = {48, 53, 58};
Spline(116) = {58, 63, 68};
Spline(117) = {68, 73, 78};
Spline(118) = {78, 83, 88};
Spline(119) = {88, 93, 98};
Symmetry {0, 1, 0, 0} {
  Duplicata { Line{65, 68, 67, 64, 69, 62, 66, 75, 74, 70, 71, 61, 63, 78, 59, 77, 58, 76, 60, 79, 56, 55, 57, 99, 53, 100, 52, 119, 54, 80, 50, 49, 98, 51, 101, 47, 118, 46, 48, 81, 44, 43, 97, 45, 41, 102, 40, 117, 82, 42, 38, 37, 96, 39, 103, 35, 34, 116, 83, 36, 32, 31, 33, 95, 29, 104, 28, 115, 30, 84, 26, 25, 27, 94, 105, 23, 22, 114, 24, 85, 20, 19, 21, 93, 17, 106, 113, 16, 86, 18, 14, 13, 15, 92, 11, 107, 10, 112, 12, 87, 8, 7, 9, 91, 108, 5, 4, 111, 88, 6, 2, 1, 3, 110, 109, 90, 89}; }
}
Symmetry {0, 0, 1, 0} {
  Duplicata { Line{65, 68, 67, 64, 69, 62, 66, 74, 75, 71, 70, 61, 124, 63, 121, 78, 122, 59, 77, 126, 129, 58, 76, 79, 60, 127, 120, 132, 56, 123, 128, 55, 125, 57, 99, 53, 100, 131, 133, 138, 52, 119, 54, 80, 50, 135, 134, 49, 142, 137, 136, 98, 51, 101, 47, 46, 118, 140, 148, 143, 48, 81, 44, 141, 43, 97, 153, 144, 145, 45, 41, 102, 146, 147, 117, 40, 152, 158, 42, 82, 38, 150, 37, 151, 96, 39, 163, 103, 35, 155, 154, 34, 116, 156, 157, 36, 83, 162, 169, 160, 32, 161, 31, 173, 165, 164, 33, 95, 167, 166, 104, 29, 172, 179, 115, 28, 170, 84, 30, 171, 174, 175, 182, 26, 25, 177, 176, 27, 180, 188, 183, 94, 23, 105, 181, 114, 22, 24, 85, 185, 184, 192, 187, 186, 20, 19, 21, 190, 193, 198, 191, 93, 17, 106, 16, 113, 195, 194, 202, 86, 18, 197, 196, 200, 14, 209, 203, 201, 13, 15, 204, 205, 206, 207, 212, 92, 107, 11, 112, 10, 12, 87, 210, 211, 218, 213, 214, 215, 217, 216, 8, 7, 9, 222, 220, 221, 91, 5, 108, 111, 4, 88, 6, 223, 229, 225, 224, 226, 227, 2, 1, 3, 232, 230, 231, 233, 234, 235, 110, 109, 90, 89}; }
}
Line Loop(458) = {70, 66, 71};
Ruled Surface(459) = {-458};
Line Loop(460) = {129, 126, 71};
Ruled Surface(461) = {460};
Line Loop(462) = {127, 120, -129};
Ruled Surface(463) = {462};
Line Loop(464) = {128, 123, -127};
Ruled Surface(465) = {464};
Line Loop(466) = {128, 265, -261};
Ruled Surface(467) = {-466};
Line Loop(468) = {256, -262, -261};
Ruled Surface(469) = {468};
Line Loop(470) = {245, 256, 255};
Ruled Surface(471) = {-470};
Line Loop(472) = {246, 242, 245};
Ruled Surface(473) = {472};
Line Loop(474) = {244, 236, -246};
Ruled Surface(475) = {474};
Line Loop(476) = {74, 239, -244};
Ruled Surface(477) = {476};
Line Loop(478) = {74, 64, -75};
Ruled Surface(479) = {-478};
Line Loop(480) = {70, -65, -75};
Ruled Surface(481) = {480};
Line Loop(482) = {78, 60, -79, -66};
Ruled Surface(483) = {-482};
Line Loop(484) = {65, 78, -59, -77};
Ruled Surface(485) = {484};
Line Loop(486) = {76, 58, -77, -64};
Ruled Surface(487) = {-486};
Line Loop(488) = {254, -257, -76, 239};
Ruled Surface(489) = {-488};
Line Loop(490) = {253, -251, -236, 254};
Ruled Surface(491) = {490};
Line Loop(492) = {260, -259, -242, 251};
Ruled Surface(493) = {492};
Line Loop(494) = {259, -275, -274, 255};
Ruled Surface(495) = {494};
Line Loop(496) = {274, -282, -281, 262};
Ruled Surface(497) = {496};
Line Loop(498) = {281, -286, -137, 265};
Ruled Surface(499) = {498};
Line Loop(500) = {137, 136, -135, -123};
Ruled Surface(501) = {500};
Line Loop(502) = {135, 134, -133, -120};
Ruled Surface(503) = {502};
Line Loop(504) = {133, 138, -79, -126};
Ruled Surface(505) = {504};
Line Loop(506) = {134, -143, -144, -145};
Ruled Surface(507) = {-506};
Line Loop(508) = {143, 138, 80, -148};
Ruled Surface(509) = {-508};
Line Loop(511) = {80, -54, 99, 60};
Ruled Surface(512) = {511};
Line Loop(513) = {99, -59, 100, 53};
Ruled Surface(514) = {-513};
Line Loop(515) = {100, -52, 119, 58};
Ruled Surface(516) = {515};
Line Loop(517) = {119, 257, 272, -276};
Ruled Surface(518) = {-517};
Line Loop(519) = {271, 270, -253, 272};
Ruled Surface(520) = {519};
Line Loop(521) = {278, -279, -260, -270};
Ruled Surface(522) = {521};
Line Loop(523) = {279, -294, 295, 275};
Ruled Surface(524) = {523};
Line Loop(525) = {295, -282, 304, 303};
Ruled Surface(526) = {-525};
Line Loop(527) = {304, -308, 147, 286};
Ruled Surface(528) = {527};
Line Loop(529) = {147, 136, 145, -146};
Ruled Surface(530) = {-529};
Line Loop(531) = {148, 81, -158, 152};
Ruled Surface(532) = {-531};
Line Loop(533) = {158, 82, -169, 162};
Ruled Surface(534) = {-533};
Line Loop(535) = {172, 169, 83, -179};
Ruled Surface(536) = {-535};
Line Loop(538) = {183, 179, 84, -188};
Ruled Surface(539) = {-538};
Line Loop(540) = {193, 188, 85, -198};
Ruled Surface(541) = {-540};
Line Loop(542) = {203, 198, 86, -209};
Ruled Surface(543) = {-542};
Line Loop(544) = {213, 209, 87, -218};
Ruled Surface(545) = {-544};
Line Loop(546) = {223, 218, 88, -229};
Ruled Surface(547) = {-546};
Line Loop(548) = {235, 229, 89};
Ruled Surface(549) = {-548};
Line Loop(550) = {89, 90, 6};
Ruled Surface(551) = {550};
Line Loop(552) = {6, -88, -12, -91};
Ruled Surface(553) = {-552};
Line Loop(554) = {92, 18, 87, -12};
Ruled Surface(555) = {554};
Line Loop(556) = {18, -86, -24, -93};
Ruled Surface(557) = {-556};
Line Loop(558) = {24, -85, -30, -94};
Ruled Surface(559) = {-558};
Line Loop(560) = {30, -84, -36, -95};
Ruled Surface(561) = {-560};
Line Loop(562) = {96, 42, 83, -36};
Ruled Surface(563) = {562};
Line Loop(564) = {97, 48, 82, -42};
Ruled Surface(565) = {564};
Line Loop(566) = {98, 54, 81, -48};
Ruled Surface(567) = {566};
Line Loop(568) = {98, -53, 101, 47};
Ruled Surface(569) = {-568};
Line Loop(570) = {97, -47, 102, 41};
Ruled Surface(571) = {-570};
Line Loop(572) = {96, -41, 103, 35};
Ruled Surface(573) = {-572};
Line Loop(574) = {104, 29, 95, -35};
Ruled Surface(575) = {-574};
Line Loop(576) = {105, 23, 94, -29};
Ruled Surface(577) = {-576};
Line Loop(578) = {106, 17, 93, -23};
Ruled Surface(579) = {-578};
Line Loop(580) = {107, 11, 92, -17};
Ruled Surface(581) = {-580};
Line Loop(582) = {108, 5, 91, -11};
Ruled Surface(583) = {-582};
Line Loop(584) = {109, 90, -5};
Ruled Surface(585) = {-584};
Line Loop(586) = {109, 110, 4};
Ruled Surface(587) = {586};
Line Loop(588) = {111, 10, 108, -4};
Ruled Surface(589) = {588};
Line Loop(590) = {112, 16, 107, -10};
Ruled Surface(591) = {590};
Line Loop(592) = {113, 22, 106, -16};
Ruled Surface(593) = {592};
Line Loop(594) = {114, 28, 105, -22};
Ruled Surface(595) = {594};
Line Loop(596) = {115, 34, 104, -28};
Ruled Surface(597) = {596};
Line Loop(598) = {116, 40, 103, -34};
Ruled Surface(599) = {598};
Line Loop(600) = {117, 46, 102, -40};
Ruled Surface(601) = {600};
Line Loop(602) = {118, 52, 101, -46};
Ruled Surface(603) = {602};
Line Loop(605) = {154, 155, 152, -144};
Ruled Surface(606) = {605};
Line Loop(607) = {165, 164, 162, -155};
Ruled Surface(608) = {607};
Line Loop(609) = {174, 175, 172, -164};
Ruled Surface(610) = {609};
Line Loop(611) = {185, 184, 183, -175};
Ruled Surface(612) = {611};
Line Loop(613) = {194, 195, 193, -184};
Ruled Surface(614) = {613};
Line Loop(615) = {205, 204, 203, -195};
Ruled Surface(616) = {615};
Line Loop(617) = {215, 214, 213, -204};
Ruled Surface(618) = {617};

Line Loop(619) = {224, 225, 223, -214};
Ruled Surface(620) = {619};
Line Loop(621) = {234, 235, -225};
Ruled Surface(622) = {621};
Line Loop(623) = {156, 146, 154, -157};
Ruled Surface(624) = {-623};
Line Loop(625) = {167, 157, 165, -166};
Ruled Surface(626) = {-625};
Line Loop(627) = {177, 166, 174, -176};
Ruled Surface(628) = {-627};
Line Loop(629) = {187, 176, 185, -186};
Ruled Surface(630) = {-629};
Line Loop(631) = {197, 186, 194, -196};
Ruled Surface(632) = {-631};
Line Loop(633) = {206, 196, 205, -207};
Ruled Surface(634) = {-633};
Line Loop(635) = {217, 207, 215, -216};
Ruled Surface(636) = {-635};
Line Loop(637) = {227, 216, 224, -226};
Ruled Surface(638) = {-637};
Line Loop(639) = {233, 226, 234};
Ruled Surface(640) = {-639};
Line Loop(641) = {452, 233, 443};
Ruled Surface(642) = {641};
Line Loop(643) = {442, -443, 227, 425};
Ruled Surface(644) = {643};
Line Loop(645) = {423, -425, 217, 409};
Ruled Surface(646) = {645};
Line Loop(647) = {407, -409, 206, 398};
Ruled Surface(648) = {647};
Line Loop(649) = {393, -398, 197, 379};
Ruled Surface(650) = {649};
Line Loop(651) = {375, -379, 187, 362};
Ruled Surface(652) = {651};
Line Loop(653) = {356, -362, 177, 345};
Ruled Surface(654) = {653};
Line Loop(655) = {340, -345, 167, 330};
Ruled Surface(656) = {655};
Line Loop(657) = {326, -330, 156, 308};
Ruled Surface(658) = {657};
Line Loop(659) = {312, -303, 326, 325};
Ruled Surface(660) = {-659};
Line Loop(661) = {333, -325, 340, 341};
Ruled Surface(662) = {-661};
Line Loop(663) = {348, -341, 356, 357};
Ruled Surface(664) = {-663};
Line Loop(665) = {366, -357, 375, 376};
Ruled Surface(666) = {-665};
Line Loop(667) = {366, 349, 353, -365};
Ruled Surface(668) = {667};
Line Loop(669) = {374, -385, 384, 365};
Ruled Surface(670) = {669};
Line Loop(671) = {395, -401, 402, 385};
Ruled Surface(672) = {671};
Line Loop(673) = {417, -420, 421, 401};
Ruled Surface(674) = {673};
Line Loop(675) = {437, -440, 439, 420};
Ruled Surface(676) = {675};
Line Loop(677) = {457, 453, 440};
Ruled Surface(678) = {677};
Line Loop(679) = {456, 438, 457};
Ruled Surface(680) = {-679};
Line Loop(681) = {455, 456, -433};
Ruled Surface(682) = {681};
Line Loop(683) = {110, 436, 455};
Ruled Surface(684) = {-683};
Line Loop(686) = {453, -441, 452};
Ruled Surface(687) = {-686};
Line Loop(689) = {432, 416, 437, -438};
Ruled Surface(690) = {-689};
Line Loop(691) = {411, 396, 417, -416};
Ruled Surface(692) = {-691};
Line Loop(693) = {387, 373, 395, -396};
Ruled Surface(694) = {-693};
Line Loop(695) = {434, 433, 432, -413};
Ruled Surface(696) = {695};
Line Loop(697) = {412, 413, 411, -388};
Ruled Surface(698) = {697};
Line Loop(699) = {389, 388, 387, -368};
Ruled Surface(700) = {699};
Line Loop(701) = {367, 354, 374, -373};
Ruled Surface(702) = {-701};
Line Loop(703) = {369, 368, 367, -347};
Ruled Surface(704) = {703};
Line Loop(705) = {346, 347, 343, -324};
Ruled Surface(706) = {705};
Line Loop(707) = {343, 331, 353, -354};
Ruled Surface(708) = {-707};
Line Loop(709) = {323, 324, 320, -306};
Ruled Surface(710) = {709};
Line Loop(711) = {320, 314, 332, -331};
Ruled Surface(712) = {-711};
Line Loop(713) = {307, 306, 301, -290};
Ruled Surface(714) = {713};
Line Loop(715) = {315, -314, 301, 296};
Ruled Surface(716) = {-715};
Line Loop(717) = {289, 290, 287, -271};
Ruled Surface(718) = {717};
Line Loop(719) = {287, 278, 297, -296};
Ruled Surface(720) = {-719};
Line Loop(721) = {393, 392, 384, -376};
Ruled Surface(722) = {-721};
Line Loop(723) = {407, 406, 402, -392};
Ruled Surface(724) = {-723};
Line Loop(725) = {423, 422, 421, -406};
Ruled Surface(726) = {-725};
Line Loop(727) = {439, -422, 442, 441};
Ruled Surface(728) = {-727};
Line Loop(729) = {312, 294, 297, -313};
Ruled Surface(730) = {729};
Line Loop(731) = {333, 313, 315, -334};
Ruled Surface(732) = {731};
Line Loop(733) = {348, 334, 332, -349};
Ruled Surface(734) = {733};
Line Loop(735) = {111, 415, 434, -436};
Ruled Surface(736) = {-735};
Line Loop(737) = {112, 390, 412, -415};
Ruled Surface(738) = {-737};
Line Loop(739) = {113, 372, 389, -390};
Ruled Surface(740) = {-739};
Line Loop(741) = {114, 351, 369, -372};
Ruled Surface(742) = {-741};
Line Loop(743) = {115, 327, 346, -351};
Ruled Surface(744) = {-743};
Line Loop(745) = {116, 311, 323, -327};
Ruled Surface(746) = {-745};
Line Loop(747) = {117, 291, 307, -311};
Ruled Surface(748) = {-747};
Line Loop(749) = {118, 276, 289, -291};
Ruled Surface(750) = {-749};
Physical Surface(1) = {518, 520, 750, 718, 516, 714, 748, 603, 491, 489, 487, 601, 710, 746, 477, 475, 479, 599, 485, 493, 514, 481, 473, 522, 706, 569, 744, 720, 597, 459, 471, 571, 716, 469, 483, 495, 573, 467, 712, 461, 465, 463, 704, 742, 595, 575, 512, 708, 524, 497, 505, 499, 567, 730, 577, 501, 503, 702, 700, 740, 526, 593, 509, 565, 732, 528, 579, 694, 563, 660, 530, 734, 507, 532, 561, 658, 668, 698, 738, 662, 591, 624, 606, 692, 581, 534, 559, 670, 656, 664, 626, 608, 557, 672, 536, 654, 666, 696, 736, 589, 555, 690, 674, 583, 628, 610, 684, 539, 585, 680, 553, 676, 652, 722, 678, 551, 682, 587, 549, 687, 642, 622, 640, 728, 724, 630, 547, 541, 612, 726, 650, 644, 638, 620, 545, 543, 646, 648, 632, 614, 636, 618, 634, 616};
"""


def cylinders(h=1.0, z=1.0, r=[0.5, 1, 1.5, 1.7], origin=(0.0, 0.0, 0.0), square=False):
    """
    Create a sequence of concentric cylindrical or cuboidal objects.

    Parameters
    ----------
    h : float
        A floating point number specifying the grid size.
    z : float
        A floating point number specifying the extrusion parameter along z-axis.
    r : increasing sequence of floats
        A sequence definying the radius of each concentric cylinder.
    origin: tuple of floats
        The centre of the base of the cylinder.
    square: boolean
        Specifies whether the cylindrical shape is a sequence of squares or circles

    Output
    -------
    grid : bempp_cl.Grid
        A structured grid.

    """
    if square:
        stub = ""
    else:
        stub = f"Point(1) = {{{origin[0]}, {origin[1]}, {origin[2]}, cl}};\n"

    for i, radius in enumerate(r):
        if square:
            stub += (
                f"Point({1 + 4 * i}) = {{{origin[0] - radius},{origin[1] - radius}, {origin[2]}, cl}};\n"
                f"Point({2 + 4 * i}) = {{{origin[0] + radius},{origin[1] - radius}, {origin[2]}, cl}};\n"
                f"Point({3 + 4 * i}) = {{{origin[0] + radius},{origin[1] + radius}, {origin[2]}, cl}};\n"
                f"Point({4 + 4 * i}) = {{{origin[0] - radius},{origin[1] + radius}, {origin[2]}, cl}};\n"
                "\n"
                f"Line({1 + 4 * i}) = {{{1 + 4 * i},{2 + 4 * i}}};\n"
                f"Line({2 + 4 * i}) = {{{2 + 4 * i},{3 + 4 * i}}};\n"
                f"Line({3 + 4 * i}) = {{{3 + 4 * i},{4 + 4 * i}}};\n"
                f"Line({4 + 4 * i}) = {{{4 + 4 * i},{1 + 4 * i}}};"
            )
        else:
            stub += (
                f"Point({2 + 4 * i}) = {{{origin[0] + radius},{origin[1]},{origin[2]},cl}};\n"
                f"Point({3 + 4 * i}) = {{{origin[0]},{origin[1] + radius},{origin[2]},cl}};\n"
                f"Point({4 + 4 * i}) = {{{origin[0] - radius},{origin[1]},{origin[2]},cl}};\n"
                f"Point({5 + 4 * i}) = {{{origin[0]},{origin[1] - radius},{origin[2]},cl}};\n"
                "\n"
                f"Circle({1 + 4 * i}) = {{{2 + 4 * i}, 1, {3 + 4 * i}}};\n"
                f"Circle({2 + 4 * i}) = {{{3 + 4 * i}, 1, {4 + 4 * i}}};\n"
                f"Circle({3 + 4 * i}) = {{{4 + 4 * i}, 1, {5 + 4 * i}}};\n"
                f"Circle({4 + 4 * i}) = {{{5 + 4 * i}, 1, {2 + 4 * i}}};\n"
            )
        stub += f"Line Loop({11 + i}) = {{{3 + 4 * i}, {4 + 4 * i}, {1 + 4 * i}, {2 + 4 * i}}};\n"
        if i == 0:
            stub += "Plane Surface(21) = {11};\n"
        else:
            stub += f"Plane Surface({21 + 3 * i}) = {{{11 + i}, {-(11 + i - 1)}}};\n"
    for i, _ in enumerate(r):
        stub += f"out[] = Extrude {{0,0,z}} {{Surface{{{21 + 3 * i}}}; Layers{{cl}};}};\n"
        stub += f"Reverse Surface{{{21 + 3 * i}}};\n"
        if i < len(r) - 1:
            stub += (
                f"Physical Surface({10 * (i + 1)}) = {{{21 + 3 * i}, out[0]}};\n"
                f"Physical Surface({10 * (i + 2) + (i + 1)}) = {{out[2], out[3], out[4], out[5]}};\n"
            )
        else:
            stub += f"Physical Surface({10 * (i + 1)}) = {{{21 + 3 * i}, out[0], out[2], out[3], out[4], out[5]}};\n"

    for i, _ in enumerate(r):
        stub += f"b() = Boundary{{Volume{{{i + 1}}};}};\n"

    geometry = f"cl = {h};\n" f"z = {z};\n" f"{stub}\n" "Mesh.Algorithm = 3;"

    return __generate_grid_from_geo_string(geometry)
