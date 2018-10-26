import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class AstrDamp(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('B_global', val=np.zeros((3,3)), units='N*s/m')

		self.add_output('Astr_damp', val=np.ones((3,3)))

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		residuals['Astr_damp'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Astr_damp']) - inputs['B_global']

	def solve_nonlinear(self, inputs, outputs):
		outputs['Astr_damp'] = np.matmul(linalg.inv(inputs['M_global'] + inputs['A_global']), inputs['B_global'])

	def linearize(self, inputs, outputs, partials):
		partials['Astr_damp', 'M_global'] = np.kron(np.identity(3),np.transpose(outputs['Astr_damp']))
		partials['Astr_damp', 'A_global'] = np.kron(np.identity(3),np.transpose(outputs['Astr_damp']))
		partials['Astr_damp', 'B_global'] = -np.identity(9)
		partials['Astr_damp', 'Astr_damp'] = np.kron(inputs['M_global'] + inputs['A_global'],np.identity(3))