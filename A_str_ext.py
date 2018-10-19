import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class AstrExt(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('dthrust_drotspeed', val=0., units='N*s/rad')
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('Astr_ext', val=np.zeros((3,1)))

	def apply_nonlinear(self, inputs, outputs, residuals):
		dthrust_drotspeed = inputs['dthrust_drotspeed'][0]
		CoG_rotor = inputs['CoG_rotor'][0]

		residuals['Astr_ext'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Astr_ext']) - np.array([[dthrust_drotspeed],[CoG_rotor * dthrust_drotspeed],[dthrust_drotspeed]])

	def solve_nonlinear(self, inputs, outputs):
		dthrust_drotspeed = inputs['dthrust_drotspeed'][0]
		CoG_rotor = inputs['CoG_rotor'][0]

		outputs['Astr_ext'] = np.matmul(linalg.inv(inputs['M_global'] + inputs['A_global']), np.array([[dthrust_drotspeed],[CoG_rotor * dthrust_drotspeed],[dthrust_drotspeed]]))