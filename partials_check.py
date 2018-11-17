import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.kdtree import KDTree

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from modeshape_hydrostiff import ModeshapeHydrostiff

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,100), \
'omega_wave': np.linspace(0.12,6.28,50)}

prob = Problem()
#np.random.seed(0)
ivc = IndepVarComp()
ivc.add_output('D_spar', val=np.random.rand(10))
ivc.add_output('tot_M_spar', val=np.random.rand())
ivc.add_output('CoG_spar', val=np.random.rand())
ivc.add_output('M_ball', val=np.random.rand())
ivc.add_output('CoG_ball', val=np.random.rand())
ivc.add_output('M_moor', val=np.random.rand())
ivc.add_output('z_moor', val=np.random.rand())
ivc.add_output('buoy_spar', val=np.random.rand())
ivc.add_output('CoB', val=np.random.rand())

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('check', ModeshapeHydrostiff(), promotes_inputs=['D_spar', 'tot_M_spar', 'CoG_spar', 'M_ball', 'CoG_ball', 'M_moor', 'z_moor', 'buoy_spar', 'CoB'], promotes_outputs=['K_hydrostatic'])

prob.setup()

check_partials_data = prob.check_partials(show_only_incorrect=True)

#partial_deriv_plot('kel','L_mode_elem', check_partials_data, binary=False)