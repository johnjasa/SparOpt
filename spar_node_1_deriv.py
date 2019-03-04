import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class SparNode1Deriv(ImplicitComponent):

	def setup(self):
		self.add_input('spar_spline_lhs', val=np.zeros((13,13)), units='m')
		self.add_input('spar_spline_rhs', val=np.zeros(13), units='m')

		self.add_output('x_d_sparnode', val=np.ones(13), units='m/m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		A = inputs['spar_spline_lhs']
		b = inputs['spar_spline_rhs']

		residuals['x_d_sparnode'] = A.dot(outputs['x_d_sparnode']) - b

	def solve_nonlinear(self, inputs, outputs):
		A = inputs['spar_spline_lhs']
		b = inputs['spar_spline_rhs']

		outputs['x_d_sparnode'] = linalg.solve(A,b)

	def linearize(self, inputs, outputs, partials):
		partials['x_d_sparnode', 'spar_spline_lhs'] = np.kron(np.identity(13),np.transpose(outputs['x_d_sparnode']))
		partials['x_d_sparnode', 'spar_spline_rhs'] = -np.identity(13)
		partials['x_d_sparnode', 'x_d_sparnode'] = inputs['spar_spline_lhs']