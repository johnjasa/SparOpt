import numpy as np
import scipy.special as ss

from openmdao.api import ExplicitComponent

class HullFatigue(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_hull_stress', val=np.zeros((N_omega,10)), units='MPa**2*s/rad')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('hull_fatigue_damage', val=np.zeros(10))

		Cols = Cols1 = np.arange(0,10*N_omega,10)
		for i in xrange(1,10):
			Cols = np.concatenate((Cols,Cols1 + i),0)

		self.declare_partials('hull_fatigue_damage', 'resp_hull_stress', rows=np.repeat(np.arange(10),N_omega), cols=Cols)
		self.declare_partials('hull_fatigue_damage', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		omega = self.omega

		resp_hull_stress = inputs['resp_hull_stress']
		wt_spar_p = inputs['wt_spar_p'][:-1]

		for i in xrange(len(wt_spar_p)):
			S_stress = resp_hull_stress[:,i]
			wt = wt_spar_p[i]

			logC = 12.164
			k = 3.0
			wt_ref = 0.025
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
			
			if wt > wt_ref:
				outputs['hull_fatigue_damage'][i] = T * C**(-1.) * v_p * (2. * sigma)**k * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3)) * (wt / wt_ref)**(t_exp * k)
			else:
				outputs['hull_fatigue_damage'][i] = T * C**(-1.) * v_p * (2. * sigma)**k * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3))

	def compute_partials(self, inputs, partials): #TODO check
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		resp_hull_stress = inputs['resp_hull_stress']
		wt_spar_p = inputs['wt_spar_p']

		for i in xrange(len(wt_spar_p)):
			S_stress = resp_hull_stress[:,i]
			wt = wt_spar_p[i][:-1]

			logC = 12.164
			k = 3.0
			wt_ref = 0.025
			t_exp = 0.2

			T = 3600. #seconds
			
			C = 10**logC

			m0 = np.trapz(S_stress,omega)
			m1 = np.trapz(omega * S_stress,omega)
			m2 = np.trapz(omega**2. * S_stress,omega)
			m4 = np.trapz(omega**4. * S_stress,omega)

			dm0_dresp = np.ones((1,N_omega)) * domega
			dm1_dresp = omega * np.ones((1,N_omega)) * domega
			dm2_dresp = omega**2. * np.ones((1,N_omega)) * domega
			dm4_dresp = omega**4. * np.ones((1,N_omega)) * domega

			dm0_dresp[0,0] += -0.5 * domega
			dm1_dresp[0,0] += -0.5 * omega[0] * domega
			dm2_dresp[0,0] += -0.5 * omega[0]**2. * domega
			dm4_dresp[0,0] += -0.5 * omega[0]**4. * domega

			dm0_dresp[0,-1] += -0.5 * domega
			dm1_dresp[0,-1] += -0.5 * omega[-1] * domega
			dm2_dresp[0,-1] += -0.5 * omega[-1]**2. * domega
			dm4_dresp[0,-1] += -0.5 * omega[-1]**4. * domega

			sigma = np.sqrt(m0)

			dsigma_dresp = 0.5 / sigma * dm0_dresp

			x_m = m1 / m0 * np.sqrt(m2 / m4)
			alpha_2 = m2 / np.sqrt(m0 * m4)
			v_p = 1. / (2. * np.pi) * np.sqrt(m4 / m2)

			dx_m_dresp = (dm1_dresp / m0 - m1 / m0**2. * dm0_dresp) * np.sqrt(m2 / m4) + m1 / m0 * 0.5 * np.sqrt(m4 / m2) * (dm2_dresp / m4 - m2 / m4**2. * dm4_dresp)
			dalpha_2_dresp = dm2_dresp / np.sqrt(m0 * m4) - 0.5 * m2 / (m0 * m4)**1.5 * (dm0_dresp * m4 + dm4_dresp * m0)
			dv_p_dresp = 0.5 * 1. / (2. * np.pi) * np.sqrt(m2 / m4) * (dm4_dresp / m2 - m4 / m2**2. * dm2_dresp)

			G1 = 2. * (x_m - alpha_2 ** 2.) / (1. + alpha_2**2.)
			R = (alpha_2 - x_m - G1**2.) / (1. - alpha_2 - G1 + G1**2.)
			G2 = (1. - alpha_2 - G1 + G1**2.) / (1. - R)
			G3 = 1. - G1 - G2
			Q = 1.25 * (alpha_2 - G3 - G2 * R) / G1

			dG1_dresp = 2. * (dx_m_dresp - 2. * alpha_2 * dalpha_2_dresp) / (1. + alpha_2**2.) - G1 / (1. + alpha_2**2.) * (2. * alpha_2 * dalpha_2_dresp)
			dR_dresp = (dalpha_2_dresp - dx_m_dresp - 2. * G1 * dG1_dresp) / (1. - alpha_2 - G1 + G1**2.) - R / (1. - alpha_2 - G1 + G1**2.) * (-dalpha_2_dresp - dG1_dresp + 2. * G1 * dG1_dresp)
			dG2_dresp = (-dalpha_2_dresp - dG1_dresp + 2. * G1 * dG1_dresp) / (1. - R) - G2 / (1. - R) * (-dR_dresp)
			dG3_dresp = -dG1_dresp - dG2_dresp
			dQ_dresp = 1.25 * (dalpha_2_dresp - dG3_dresp - dG2_dresp * R - G2 * dR_dresp) / G1 - Q / G1 * dG1_dresp

			if wt > wt_ref:
				partials['hull_fatigue_damage', 'resp_hull_stress'][i*N_omega:i*N_omega+N_omega] = T * C**(-1.) * (dv_p_dresp * (2. * sigma)**k + v_p * k * (2. * sigma)**(k - 1.) * 2. * dsigma_dresp) * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3)) * (wt / wt_ref)**(t_exp * k) + T * C**(-1.) * v_p * (2. * sigma)**k * (ss.gamma(1. + k) * (dG1_dresp * Q**k + G1 * k * Q**(k - 1.) * dQ_dresp) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (dG2_dresp * R**k + G2 * k * R**(k - 1.) * dR_dresp + dG3_dresp)) * (wt / wt_ref)**(t_exp * k)
				partials['hull_fatigue_damage', 'wt_spar_p'][i] = T * C**(-1.) * v_p * (2. * sigma)**k * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3)) * (t_exp * k / wt_ref) * (wt / wt_ref)**(t_exp * k - 1.)
			else:
				partials['hull_fatigue_damage', 'resp_hull_stress'][i*N_omega:i*N_omega+N_omega] = T * C**(-1.) * (dv_p_dresp * (2. * sigma)**k + v_p * k * (2. * sigma)**(k - 1.) * 2. * dsigma_dresp) * (G1 * Q**k * ss.gamma(1. + k) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (G2 * R**k + G3)) + T * C**(-1.) * v_p * (2. * sigma)**k * (ss.gamma(1. + k) * (dG1_dresp * Q**k + G1 * k * Q**(k - 1.) * dQ_dresp) + np.sqrt(2.)**k * ss.gamma(1. + k / 2.) * (dG2_dresp * R**k + G2 * k * R**(k - 1.) * dR_dresp + dG3_dresp))
				partials['hull_fatigue_damage', 'wt_spar_p'][i] = 0.