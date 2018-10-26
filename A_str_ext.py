import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class AstrExt(ImplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('dthrust_drotspeed', val=0., units='N*s/rad')
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('Astr_ext', val=np.ones((3,1)))

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		dthrust_drotspeed = inputs['dthrust_drotspeed'][0]
		CoG_rotor = inputs['CoG_rotor'][0]

		residuals['Astr_ext'] = (inputs['M_global'] + inputs['A_global']).dot(outputs['Astr_ext']) - np.array([[dthrust_drotspeed],[CoG_rotor * dthrust_drotspeed],[dthrust_drotspeed]])

	def solve_nonlinear(self, inputs, outputs):
		dthrust_drotspeed = inputs['dthrust_drotspeed'][0]
		CoG_rotor = inputs['CoG_rotor'][0]

		outputs['Astr_ext'] = np.matmul(linalg.inv(inputs['M_global'] + inputs['A_global']), np.array([[dthrust_drotspeed],[CoG_rotor * dthrust_drotspeed],[dthrust_drotspeed]]))

	def linearize(self, inputs, outputs, partials):
		dthrust_drotspeed = inputs['dthrust_drotspeed'][0]
		CoG_rotor = inputs['CoG_rotor'][0]
		
		partials['Astr_ext', 'M_global'] = np.kron(np.identity(3),np.transpose(outputs['Astr_ext']))
		partials['Astr_ext', 'A_global'] = np.kron(np.identity(3),np.transpose(outputs['Astr_ext']))
		partials['Astr_ext', 'dthrust_drotspeed'] = -np.array([[1.],[CoG_rotor],[1.]])
		partials['Astr_ext', 'CoG_rotor'] = -np.array([[0.],[dthrust_drotspeed],[0.]])
		partials['Astr_ext', 'Astr_ext'] = inputs['M_global'] + inputs['A_global']