import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from wave_loads import WaveLoads

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,3493), \
'omega_wave': 2. * np.pi / np.linspace(40.,1.,80)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('x_sparelem', val=np.random.rand(13)*5.)
ivc.add_output('z_sparnode', val=np.linspace(-120,11,14))
ivc.add_output('D_spar', val=np.random.rand(10)*10.)
ivc.add_output('Z_spar', val=np.linspace(-120,10,11))
ivc.add_output('water_depth', val=320.)
ivc.add_output('wave_number', val=np.array([3.23990791e-03, 3.29306619e-03, 3.34816990e-03, 3.40531892e-03, 3.46462543e-03, 3.52603769e-03, 3.59023551e-03, 3.65688664e-03, 3.72619898e-03, 3.79837885e-03, 3.87363521e-03, 3.95218414e-03, 4.03425326e-03, 4.12008598e-03, 4.20994535e-03, 4.30411760e-03, 4.40291540e-03, 4.50651742e-03, 4.61584067e-03, 4.73066825e-03, 4.85160629e-03, 4.97924157e-03, 5.11416115e-03, 5.25697221e-03, 5.40832050e-03, 5.56890665e-03, 5.73949985e-03, 5.92094881e-03, 6.11419072e-03, 6.32025876e-03, 6.54028888e-03, 6.77552692e-03, 7.02733636e-03, 7.29664760e-03, 7.58654859e-03, 7.89774549e-03, 8.23223977e-03, 8.59167863e-03, 8.98013398e-03, 9.39847886e-03, 9.85014853e-03, 1.03383296e-02, 1.08665881e-02, 1.14389409e-02, 1.20599403e-02, 1.27342993e-02, 1.34690656e-02, 1.42704630e-02, 1.51464440e-02, 1.61062633e-02, 1.71607343e-02, 1.83225434e-02, 1.96066384e-02, 2.10307139e-02, 2.26158240e-02, 2.43871632e-02, 2.63750713e-02, 2.86163389e-02, 3.11559194e-02, 3.40492003e-02, 3.73650475e-02, 4.11899416e-02, 4.56336737e-02, 5.08373137e-02, 5.69845450e-02, 6.43181004e-02, 7.31640961e-02, 8.39689246e-02, 9.73566943e-02, 1.14221421e-01, 1.35880249e-01, 1.64338655e-01, 2.02771968e-01, 2.56451101e-01, 3.34650993e-01, 4.54943557e-01, 6.54005049e-01, 1.01928102e+00, 1.80438509e+00, 4.02567825e+00]))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('check', WaveLoads(freqs=freqs), promotes_inputs=['x_sparelem', 'z_sparnode', 'D_spar', 'Z_spar', 'water_depth', 'wave_number'], promotes_outputs=['Re_wave_forces', 'Im_wave_forces'])

prob.setup()

check_partials_data = prob.check_partials()

#partial_deriv_plot('Re_wave_forces', 'wave_number', check_partials_data, binary=False)