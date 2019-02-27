import numpy as np

from openmdao.api import ExplicitComponent

class StdDevTowerMoment(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_tower_moment', val=np.zeros((N_omega,11)), units='(N*m)**2*s/rad')

		self.add_output('stddev_tower_moment', val=np.zeros(11), units='N*m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		for i in xrange(11):
			outputs['stddev_tower_moment'][i] = np.sqrt(np.trapz(inputs['resp_tower_moment'][:,i], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_tower_moment', 'resp_tower_moment'] = np.zeros((11,11*N_omega))
		
		for i in xrange(11):
			partials['stddev_tower_moment', 'resp_tower_moment'][i,i:11*N_omega:11] += np.ones(N_omega) * 0.5 / np.sqrt(np.trapz(inputs['resp_tower_moment'][:,i], omega)) * domega

			partials['stddev_tower_moment', 'resp_tower_moment'][i,i] += -0.5 / np.sqrt(np.trapz(inputs['resp_tower_moment'][:,i], omega)) * domega / 2.
			partials['stddev_tower_moment', 'resp_tower_moment'][i,11*N_omega-11+i] += -0.5 / np.sqrt(np.trapz(inputs['resp_tower_moment'][:,i], omega)) * domega / 2.