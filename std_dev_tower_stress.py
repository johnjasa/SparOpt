import numpy as np

from openmdao.api import ExplicitComponent

class StdDevTowerStress(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_tower_stress', val=np.zeros((N_omega,11)), units='MPa**2*s/rad')

		self.add_output('stddev_tower_stress', val=np.zeros(10), units='MPa')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		for i in xrange(10):
			outputs['stddev_tower_stress'][i] = np.sqrt(np.trapz(inputs['resp_tower_stress'][:,i], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_tower_stress', 'resp_tower_stress'] = np.zeros((10,10*N_omega))
		
		for i in xrange(10):
			partials['stddev_tower_stress', 'resp_tower_stress'][i,i:10*N_omega:10] += np.ones(N_omega) * 0.5 / np.sqrt(np.trapz(inputs['resp_tower_stress'][:,i], omega)) * domega

			partials['stddev_tower_stress', 'resp_tower_stress'][i,i] += -0.5 / np.sqrt(np.trapz(inputs['resp_tower_stress'][:,i], omega)) * domega / 2.
			partials['stddev_tower_stress', 'resp_tower_stress'][i,10*N_omega-10+i] += -0.5 / np.sqrt(np.trapz(inputs['resp_tower_stress'][:,i], omega)) * domega / 2.