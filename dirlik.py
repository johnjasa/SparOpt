import numpy as np

from openmdao.api import ExplicitComponent

class Dirlik(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_wave_forces', val=np.zeros((80,3,1)))
		self.add_input('Im_wave_forces', val=np.zeros((80,3,1)))
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))

		self.add_output('fatigue_damage', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		S_stress = inputs['']

		logC = 12.164
		k = 3.0
		t_ref = 0.025
		t_exp = 0.2

		T = 3600. #seconds
		
		C = 10**logC

		m0 = np.trapz(S_stress,omega)
		m1 = np.trapz(omega * S_stress,omega)
		m2 = np.trapz(omega**2. * S_stress,omega)
		m4 = np.trapz(omega**4. * S_stress,omega)

		sigma = np.sqrt(m0)

		x_m = m1 / m0 * np.sqrt(m2 / m4)
		alpha_2 = m2 / np.sqrt(m0 * m4)
		v_p = 1. / (2. * np.pi) * np.sqrt(m4 / m2)
		
		G1 = 2. * (x_m - alpha_2**2.) / (1. + alpha_2**2.)
		R = (alpha_2 - x_m - G1**2.) / (1. - alpha_2 - G1 + G1**2.)
		G2 = (1. - alpha_2 - G1 + G1**2.) / (1. - R)
		G3 = 1. - G1 - G2
		Q = 1.25 * (alpha_2 - G3 - G2 * R) / G1
		
		D = C**(-1.) * v_p * (2. * sigma)**k * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3)) * (0.038 / wt_ref)**(t_exp * k)
		
		fatigue = D * T