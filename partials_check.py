import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from modeshape_elem_length import ModeshapeElemLength

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('z_sparnode', val=np.linspace(-120,10,14))
ivc.add_output('z_towernode', val=np.linspace(10,119,11))

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('check', ModeshapeElemLength(), promotes_inputs=['z_sparnode', 'z_towernode'], promotes_outputs=['L_mode_elem'])

prob.setup()

#check_partials_data = prob.check_partials(show_only_incorrect=True)

check_partials_data = prob.check_partials(out_stream=None)
partial_deriv_plot('L_mode_elem', 'z_towernode', check_partials_data, binary=False)