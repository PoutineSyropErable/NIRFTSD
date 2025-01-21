import pyvista as pv
cube = pv.Cube()
p = pv.Plotter()
p.add_mesh(cube)
p.show()

