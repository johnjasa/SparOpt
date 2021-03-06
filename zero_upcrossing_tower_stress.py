import numpy as np

from openmdao.api import ExplicitComponent

class ZeroUpcrossingTowerStress(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_tower_stress', val=np.zeros((N_omega,11)), units='MPa**2*s/rad')

		self.add_output('v_z_tower_stress', val=np.zeros(10), units='1/s')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		S_stress = inputs['resp_tower_stress']

		for i in xrange(10):
			m0 = np.trapz(S_stress[:,i],omega)
			m2 = np.trapz(omega**2. * S_stress[:,i],omega)
			
			outputs['v_z_tower_stress'][i] = 1. / (2. * np.pi) * np.sqrt(m2 / m0)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		S_stress = inputs['resp_tower_stress']

		partials['v_z_tower_stress', 'resp_tower_stress'] = np.zeros((10,11*N_omega))

		for i in xrange(10):
			m0 = np.trapz(S_stress[:,i],omega)
			m2 = np.trapz(omega**2. * S_stress[:,i],omega)

			dm0_dresp = np.ones(N_omega) * domega
			dm2_dresp = omega**2. * domega

			dm0_dresp[0] += -domega / 2.
			dm2_dresp[0] += -omega[0]**2. * domega / 2.
			dm0_dresp[-1] += -domega / 2.
			dm2_dresp[-1] += -omega[-1]**2. * domega / 2.
		
			partials['v_z_tower_stress', 'resp_tower_stress'][i,i:11*N_omega:11] += 1. / (2. * np.pi) * 0.5 / np.sqrt(m2 / m0) * (dm2_dresp / m0 - m2 / m0**2. * dm0_dresp)
