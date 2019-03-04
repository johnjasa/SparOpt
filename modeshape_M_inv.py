import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeMInv(ExplicitComponent):

	def setup(self):
		self.add_input('M_mode', val=np.zeros((46,46)), units='kg')

		self.add_output('M_mode_inv', val=np.zeros((46,46)), units='1/kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['M_mode_inv'] = np.linalg.inv(inputs['M_mode'])

	def compute_partials(self, inputs, partials):
		M = inputs['M_mode']

		partials['M_mode_inv', 'M_mode'] = np.kron(np.linalg.inv(M),-np.linalg.inv(M).T)