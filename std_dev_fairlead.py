import numpy as np

from openmdao.api import ExplicitComponent

class StdDevFairlead(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_fairlead', val=np.zeros(N_omega), units='m**2*s/rad')

		self.add_output('stddev_fairlead', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_fairlead'] = np.sqrt(np.trapz(inputs['resp_fairlead'], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_fairlead', 'resp_fairlead'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_fairlead'], omega)) * domega
		partials['stddev_fairlead', 'resp_fairlead'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_fairlead'], omega)) * domega / 2.
		partials['stddev_fairlead', 'resp_fairlead'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_fairlead'], omega)) * domega / 2.