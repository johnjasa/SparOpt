import numpy as np
from scipy import linalg

from openmdao.api import ExplicitComponent

class TransferFunctionPre(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('A_feedbk', val=np.zeros((11,11)))

		self.add_output('Re_IA', val=np.repeat(np.identity(11)[np.newaxis, :, :], N_omega, axis=0))
		self.add_output('Im_IA', val=np.repeat(np.identity(11)[np.newaxis, :, :], N_omega, axis=0))

		self.declare_partials('Re_IA', 'A_feedbk', rows=np.arange(N_omega*11*11), cols=np.tile(np.arange(11*11),N_omega))

	def compute(self, inputs, outputs):
		omega = self.omega
		N_omega = len(omega)

		for i in xrange(N_omega):
			IA = 1j*omega[i] * np.identity(11) - inputs['A_feedbk']

			outputs['Re_IA'][i] = np.real(IA)
			outputs['Im_IA'][i] = np.imag(IA)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		partials['Re_IA', 'A_feedbk'] = -np.ones(N_omega*11*11)