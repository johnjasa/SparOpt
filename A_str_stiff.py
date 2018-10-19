import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class AstrStiff(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('K_global', val=np.zeros((3,3)), units='N/m')

		self.add_output('Astr_stiff', val=np.zeros((3,3)))

	def apply_nonlinear(self, inputs, outputs, residuals):
		residuals['Astr_stiff'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Astr_stiff']) - inputs['K_global']

	def solve_nonlinear(self, inputs, outputs):
		outputs['Astr_stiff'] = np.matmul(linalg.inv(inputs['M_global'] + inputs['A_global']), inputs['K_global'])