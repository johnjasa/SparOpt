import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeEigmatrix(ExplicitComponent):

	def setup(self):
		self.add_input('K_mode', val=np.zeros((34,34)), units='N/m')
		self.add_input('M_mode_inv', val=np.zeros((34,34)), units='1/kg')

		self.add_output('A_eig', val=np.zeros((34,34)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		K = inputs['K_mode']
		M_inv = inputs['M_mode_inv']
		
		outputs['A_eig'] = np.matmul(M_inv,K)

	def compute_partials(self, inputs, partials):
		K = inputs['K_mode']
		M_inv = inputs['M_mode_inv']

		partials['A_eig', 'K_mode'] = np.kron(M_inv,np.identity(34))
		partials['A_eig', 'M_mode_inv'] = np.kron(np.identity(34),K.T)