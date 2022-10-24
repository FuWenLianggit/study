import numpy as np
import PyDealII.Release as dealii


def refine_around_points(tria, points, radius, repeat=1, exclude_materials=[]):
    '''
        Refine cells around points Repeat several times if requested.
        Only cells that are within a given radius are refined.
    '''

    dim = tria.dim()

    for i in range(repeat):
        for cell in tria.active_cells():
            if cell.material_id in exclude_materials:
                continue

            center = cell.center().to_list()
            for point in points:
                dist = 0
                for d in range(dim):
                    dist += abs(center[d] - point[d])**2
                dist = np.sqrt(dist)

                if dist <= radius:
                    cell.refine_flag = 'isotropic'

        tria.execute_coarsening_and_refinement()


dx = np.array([150, 120, 100, 70, 50, 50, 50, 50, 70, 100, 120, 150])
dy = np.array([150, 120, 100, 70, 50, 50, 50, 50, 70, 100, 120, 150])
dz = np.array([50, 50, 50, 50, 70, 100, 120, 150])

p_origin = dealii.Point([-np.sum(dx) / 2.0, -np.sum(dy) / 2.0, 0.0])
p_end = dealii.Point([np.sum(dx) / 2.0, np.sum(dy) / 2.0, np.sum(dz)*1.0])

tria = dealii.Triangulation('3D')
tria.generate_subdivided_steps_hyper_rectangle(
    [dx.tolist(), dy.tolist(), dz.tolist()], p_origin, p_end, False)

for cell in tria.active_cells():
    cell.material_id = 0

source = [0, 0, 0]
sites = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100]

points = []
points.append(source)
for x in sites:
    points.append([x, 0, 0])

radius = 50
while radius > 0.1:
    refine_around_points(tria, points, radius, 1)
    radius /= 2

tria.refine_global(1)

tria.save('halfspace.tria')
tria.write('halfspace.vtu', 'vtu')

rhos = [100]
with open('halfspace.rho', 'w') as rhof:
    print('%d' % (len(rhos)), file=rhof)
    for rho in rhos:
        print('%g' % (rho), file=rhof)

with open('halfspace.emd', 'w') as emdf:
    print('%g %g %g' % (source[0], source[1], source[2]), file=emdf)

    print('%d' % (len(sites)), file=emdf)
    for x in sites:
        print('%g 0 0' % (x), file=emdf)
