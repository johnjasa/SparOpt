import numpy as np

from openmdao.api import ImplicitComponent

class OpPointSpar(ImplicitComponent):

	def setup(self):
		self.add_input('K_global', val=np.zeros((3,3)), units='N/m')
		self.add_input('thrust_0', val=0., units='N')
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('spar_mean', val=np.zeros((3,1)), units='m')

	def apply_nonlinear(self, inputs, outputs, residuals):
		K_global = inputs['K_global']
		thrust_0 = inputs['thrust_0']
		CoG_rotor = inputs['CoG_rotor']

		residuals['spar_mean'] = K_global.dot(outputs['spar_mean']) - np.array([[thrust_0],[CoG_rotor * thrust_0],[thrust_0]])

	def solve_nonlinear(self, inputs, outputs):
		K_global = inputs['K_global']
		thrust_0 = inputs['thrust_0']
		CoG_rotor = inputs['CoG_rotor']

		outputs['spar_mean'] = np.linalg.solve(K_global, np.array([[thrust_0],[CoG_rotor * thrust_0],[thrust_0]]))