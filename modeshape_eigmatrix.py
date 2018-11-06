import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeEigmatrix(ExplicitComponent):

	def setup(self):
		self.add_input('K_mode', val=np.zeros((34,34)), units='N/m')
		self.add_input('M_mode', val=np.zeros((34,34)), units='kg')
		self.add_input('omega_eig', val=0., units='rad/s')

		self.add_output('eig_matrix', val=np.zeros((34,34)), units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		K = inputs['K_mode']
		M = inputs['M_mode']
		
		outputs['eig_matrix'] = K - inputs['omega_eig']**2. * M

	def compute_partials(self, inputs, partials):
		K = inputs['K_mode']
		M = inputs['M_mode']

		partials['eig_matrix', 'K_mode'] = np.identity((1156))
		partials['eig_matrix', 'M_mode'] = -inputs['omega_eig']**2. * np.identity((1156))
		partials['eig_matrix', 'omega_eig'] = -2. * inputs['omega_eig'] * M.flatten()