import numpy as np

from openmdao.api import ExplicitComponent

class StdDevHullMoment(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_hull_moment', val=np.zeros((N_omega,10)), units='(N*m)**2*s/rad')

		self.add_output('stddev_hull_moment', val=np.zeros(10), units='N*m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		for i in xrange(10):
			outputs['stddev_hull_moment'][i] = np.sqrt(np.trapz(inputs['resp_hull_moment'][:,i], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_hull_moment', 'resp_hull_moment'] = np.zeros((10,10*N_omega))
		
		for i in xrange(10):
			partials['stddev_hull_moment', 'resp_hull_moment'][i,i:10*N_omega:10] += np.ones(N_omega) * 0.5 / np.sqrt(np.trapz(inputs['resp_hull_moment'][:,i], omega)) * domega

			partials['stddev_hull_moment', 'resp_hull_moment'][i,i] += -0.5 / np.sqrt(np.trapz(inputs['resp_hull_moment'][:,i], omega)) * domega / 2.
			partials['stddev_hull_moment', 'resp_hull_moment'][i,10*N_omega-10+i] += -0.5 / np.sqrt(np.trapz(inputs['resp_hull_moment'][:,i], omega)) * domega / 2.