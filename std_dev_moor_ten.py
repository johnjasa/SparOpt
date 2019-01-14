import numpy as np

from openmdao.api import ExplicitComponent

class StdDevMoorTen(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_moor_ten', val=np.zeros(N_omega), units='N**2*s/rad')

		self.add_output('stddev_moor_ten', val=0., units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_moor_ten'] = np.sqrt(np.trapz(inputs['resp_moor_ten'], omega))

	def compute_partials(self, inputs, partials): #TODO check
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_moor_ten', 'resp_moor_ten'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_moor_ten'], omega)) * domega
		partials['stddev_moor_ten', 'resp_moor_ten'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_moor_ten'], omega)) * domega / 2.
		partials['stddev_moor_ten', 'resp_moor_ten'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_moor_ten'], omega)) * domega / 2.