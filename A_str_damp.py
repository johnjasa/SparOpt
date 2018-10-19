import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class AstrDamp(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('B_global', val=np.zeros((3,3)), units='N*s/m')

		self.add_output('Astr_damp', val=np.zeros((3,3)))

	def apply_nonlinear(self, inputs, outputs, residuals):
		residuals['Astr_damp'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Astr_damp']) - inputs['B_global']

	def solve_nonlinear(self, inputs, outputs):
		outputs['Astr_damp'] = np.matmul(linalg.inv(inputs['M_global'] + inputs['A_global']), inputs['B_global'])