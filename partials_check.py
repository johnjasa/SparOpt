import numpy as np
import matplotlib.pyplot as plt

from openmdao.api import Problem, IndepVarComp, DirectSolver, NewtonSolver
from openmdao.utils.visualization import partial_deriv_plot

from modeshape_eigvector import ModeshapeEigvector

freqs = {\
'omega' : np.linspace(0.014361566416410483,6.283185307179586,100), \
'omega_wave': np.linspace(0.12,6.28,50)}
"""
N = 48
M = np.diag(range(1,N+1))
np.random.seed(3)
K = np.random.rand(N,N)
K = (K + K.T)/2

A = np.linalg.solve(M,K)

prob = Problem()
ivc = IndepVarComp()
ivc.add_output('A_eig', val=A)

prob.model.add_subsystem('prob_vars', ivc, promotes=['*'])

comp = prob.model.add_subsystem('check', ModeshapeEigvector(), promotes_inputs=['A_eig'], promotes_outputs=['eig_vector'])

prob.setup(force_alloc_complex=True)
comp.set_check_partial_options(wrt='*', step=1e-8, method='cs')
check_partials_data = prob.check_partials(show_only_incorrect=True)
"""
#partial_deriv_plot('eig_vector','A_eig', check_partials_data, binary=True)