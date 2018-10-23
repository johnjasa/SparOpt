import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

from openmdao.api import Problem, IndepVarComp, DirectSolver, NonlinearBlockGS
from openmdao.utils.visualization import partial_deriv_plot

from C_contrl import Ccontrl

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('k_i', val=0.016)
ivc.add_output('k_p', val=0.17)
ivc.add_output('gain_corr_factor', val=0.34)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

prob.model.add_subsystem('C_contrl', Ccontrl(), promotes_inputs=['k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['C_contrl'])

prob.setup()

#check_partials_data = prob.check_partials(show_only_incorrect=True)

check_partials_data = prob.check_partials(out_stream=None)
partial_deriv_plot('C_contrl', 'k_i', check_partials_data, binary=False)