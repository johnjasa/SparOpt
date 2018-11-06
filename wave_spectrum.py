import numpy as np

from openmdao.api import ExplicitComponent

class WaveSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)
		
		self.add_input('Hs', val=0., units='m')
		self.add_input('Tp', val=0., units='s')
		#self.add_input('omega', val=np.zeros(3493), units='rad/s')

		self.add_output('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')

	def compute(self, inputs, outputs):
		omega = self.omega#inputs['omega']
		N_omega = len(omega)

		gamma = 3.3

		Tz = inputs['Tp'] / (1.407 * (1. - 0.287 * np.log(gamma))**0.25)

		alpha = 1.2905 * inputs['Hs']**2. / Tz**4.
		beta = 1.25

		omega_p = 2. * np.pi / inputs['Tp']

		for i in xrange(N_omega):
			sigma = 0.07
			if omega[i] > omega_p:
				sigma = 0.09
		
			outputs['S_wave'][i] = alpha * 9.80665**2. * omega[i]**(-5.) * np.exp(-beta * (omega_p / omega[i])**4.) * gamma**np.exp(-((omega[i] - omega_p)**2.) / (2. * sigma**2. * omega_p**2.))