import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from modeshape_M_inv import ModeshapeMInv

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,3493), \
'omega_wave': 2. * np.pi / np.linspace(40.,1.,80)}

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('M_mode', val=np.random.rand(34,34))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('check', ModeshapeMInv(), promotes_inputs=['M_mode'], promotes_outputs=['M_mode_inv'])

prob.setup()

check_partials_data = prob.check_partials()

#partial_deriv_plot('M_mode_inv', 'M_mode', check_partials_data, binary=False)