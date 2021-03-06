import numpy as np

from openmdao.api import ExplicitComponent

class WindSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)
		
		self.add_input('windspeed_0', val=0., units='m/s')
		#self.add_input('omega', val=np.zeros(3493), units='rad/s')

		self.add_output('S_wind', val=np.ones(N_omega), units='m**2/(rad*s)')

	def compute(self, inputs, outputs):
		omega = self.omega#inputs['omega']
		Vhub = inputs['windspeed_0']

		N_omega = len(omega)

		Iref = 0.14 #Assumes NTM and wind turbine class B

		sigma = Iref * (0.75 * Vhub + 5.6) #Turbulence standard deviation

		Lambda_U = 42.0 #assumes that hub height is larger than 60m

		L = 8.1 * Lambda_U

		for i in xrange(N_omega):
			freq = omega[i] / (2. * np.pi)
			outputs['S_wind'][i] = (4. * sigma**2. * L / Vhub) / ((1. + 6. * freq * L / Vhub)**(5./3.)) / (2. * np.pi)