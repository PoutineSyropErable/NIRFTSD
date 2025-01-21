import pyvista as pv
sphere = pv.Sphere()
p = pv.Plotter()
p.add_mesh(sphere)
p.show()

