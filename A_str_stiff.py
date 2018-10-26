import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class AstrStiff(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('K_global', val=np.zeros((3,3)), units='N/m')

		self.add_output('Astr_stiff', val=np.ones((3,3)))

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		residuals['Astr_stiff'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Astr_stiff']) - inputs['K_global']

	def solve_nonlinear(self, inputs, outputs):
		outputs['Astr_stiff'] = np.matmul(linalg.inv(inputs['M_global'] + inputs['A_global']), inputs['K_global'])

	def linearize(self, inputs, outputs, partials):
		partials['Astr_stiff', 'M_global'] = np.kron(np.identity(3),np.transpose(outputs['Astr_stiff']))
		partials['Astr_stiff', 'A_global'] = np.kron(np.identity(3),np.transpose(outputs['Astr_stiff']))
		partials['Astr_stiff', 'K_global'] = -np.identity(9)
		partials['Astr_stiff', 'Astr_stiff'] = np.kron(inputs['M_global'] + inputs['A_global'],np.identity(3))