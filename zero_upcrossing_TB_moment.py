import numpy as np

from openmdao.api import ExplicitComponent

class ZeroUpcrossingTBMoment(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_TB_moment', val=np.zeros(N_omega))

		self.add_output('Nz_TB_moment', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		S_moment = inputs['resp_TB_moment']

		T = 3600. #seconds

		m0 = np.trapz(S_moment,omega)
		m2 = np.trapz(omega**2. * S_moment,omega)
		
		v_z = 1. / (2. * np.pi) * np.sqrt(m2 / m0)

		outputs['Nz_TB_moment'] = T * v_z
		

	def compute_partials(self, inputs, partials): #TODO check
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		S_moment = inputs['resp_TB_moment']

		T = 3600. #seconds

		m0 = np.trapz(S_moment,omega)
		m2 = np.trapz(omega**2. * S_moment,omega)

		dm0_dresp = np.ones((1,N_omega)) * domega
		dm2_dresp = omega**2. * np.ones((1,N_omega)) * domega
		
		partials['Nz_TB_moment', 'resp_TB_moment'] = T * 1. / (2. * np.pi) * 0.5 / np.sqrt(m2 / m0) * (dm2_dresp / m0 - m2 / m0**2. * dm0_dresp)
