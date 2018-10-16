import numpy as np

from openmdao.api import ExplicitComponent

class WindSpectrum(ExplicitComponent):

	def setup(self):
		self.add_input('Vhub', val=0., units='m/s')
		self.add_input('omega', val=np.zeros(N_omega), units='rad/s')

		self.add_output('S_wind', val=np.zeros(N_omega), units='m**2*s/rad')

	def compute(self, inputs, outputs):
		omega = inputs['omega']
		Vhub = inputs['Vhub']

		Iref = 0.14 #Assumes NTM and wind turbine class B

		sigma = Iref * (0.75 * Vhub + 5.6) #Turbulence standard deviation

		Lambda_U = 42.0 #assumes that hub height is larger than 60m

		L = 8.1 * Lambda_U

		for i in xrange(N_omega):
			omega[i] / (2. * np.pi)
			outputs['S_wind'][i] = (4. * sigma**2. * L / Vhub) / ((1. + 6. * freq * L / Vhub)**(5./3.)) / (2. * np.pi)