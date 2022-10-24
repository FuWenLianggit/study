import numpy as np
import PyDealII.Release as dealii

dx = np.array([150, 120, 100, 70, 50, 50, 50, 50, 70, 100, 120, 150])
dy = np.array([150, 120, 100, 70, 50, 50, 50, 50, 70, 100, 120, 150])
dz = np.array([50, 50, 50, 50, 70, 100, 120, 150])

p_origin = dealii.Point([-np.sum(dx) / 2.0, -np.sum(dy) / 2.0, 0.0])
p_end = dealii.Point([np.sum(dx) / 2.0, np.sum(dy) / 2.0, np.sum(dz)*1.0])

tria = dealii.Triangulation('3D')
tria.generate_subdivided_steps_hyper_rectangle(
    [dx.tolist(), dy.tolist(), dz.tolist()], p_origin, p_end, False)

for cell in tria.active_cells():
    c = cell.center().to_list()
    if c[0] > -50 and c[0] < 50 and c[1] > -50 and c[1] < 50 and c[2] > 50 and c[2] < 150:
        cell.material_id = 1
    else:
        cell.material_id = 0

tria.refine_global(2)

tria.save('block.tria')
tria.write('block.vtu', 'vtu')

rhos = [100, 1]
with open('block.rho', 'w') as rhof:
    print('%d' % (len(rhos)), file=rhof)
    for rho in rhos:
        print('%g' % (rho), file=rhof)

source = [0, 0, 0]
sites = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100]

with open('block.emd', 'w') as emdf:
    print('%g %g %g' % (source[0], source[1], source[2]), file=emdf)

    print('%d' % (len(sites)), file=emdf)
    for x in sites:
        print('%g 0 0' % (x), file=emdf)
