import numpy as np
import scipy.special as ss
import scipy.optimize as so
import scipy.interpolate as si

from openmdao.api import ExplicitComponent

class WaveLoads(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_sparnode', val=np.zeros(14), units='m')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('omega_wave', val=np.zeros(80), units='rad/s')
		self.add_input('water_depth', val=0., units='m')

		self.add_output('Re_wave_forces', val=np.zeros((80,3,1)))
		self.add_output('Im_wave_forces', val=np.zeros((80,3,1)))

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		omega_wave = inputs['omega_wave']
		h = inputs['water_depth'][0]

		f_psi_spar = si.UnivariateSpline(inputs['z_sparnode'], inputs['x_sparnode'], s=0)

		outputs['Re_wave_forces'] = np.zeros((80,3,1))
		outputs['Im_wave_forces'] = np.zeros((80,3,1))

		wavenum = np.zeros(80)

		def F(x): #define implicit equation for wave number
			return x*9.80665*np.tanh(x*h) - omega_wave[i]**2.

		for i in xrange(80):
			wavenum = so.broyden1(F,1.01*omega_wave[i]**2./9.80665)
			if wavenum < 0.0:
				wavenum = -wavenum
			if (wavenum * h) > 710: #upper limit for numpy cosh function (returns inf for larger numbers)
				continue

			for j in xrange(len(D_spar)):
				Nelem = 2
				a = D_spar[j] / 2.

				if j == len(D_spar) - 1:
					L_elem = (L_spar[j] - 10.) / Nelem
				else:
					L_elem = L_spar[j] / Nelem

				for k in xrange(Nelem):
					if j == len(D_spar) - 1:
						z = -3. + 2 * k
					else:
						z = -120. + np.sum(L_spar[0:j]) + L_elem * (k + 0.5)
					J = ss.jvp(1,wavenum*a,1)
					Y = ss.yvp(1,wavenum*a,1)
					G = 1. / np.sqrt(J**2. + Y**2.)
					alpha = np.arctan2(J,Y)
			
					X1 = 4. * 1025. * 9.80665 / wavenum * np.cosh(wavenum * (z + h)) / np.cosh(wavenum * h) * G * L_elem * np.exp(1j * alpha + 1j * np.pi / 2.)
					X5 = 4. * 1025. * 9.80665 / wavenum * np.cosh(wavenum * (z + h)) / np.cosh(wavenum * h) * G * L_elem * z * np.exp(1j * alpha + 1j * np.pi / 2.)
					X7 = 4. * 1025. * 9.80665 / wavenum * np.cosh(wavenum * (z + h)) / np.cosh(wavenum * h) * G * L_elem * f_psi_spar(z) * np.exp(1j * alpha + 1j * np.pi / 2.)

					outputs['Re_wave_forces'][i,0,0] += np.real(X1)
					outputs['Re_wave_forces'][i,1,0] += np.real(X5)
					outputs['Re_wave_forces'][i,2,0] += np.real(X7)

					outputs['Im_wave_forces'][i,0,0] += np.imag(X1)
					outputs['Im_wave_forces'][i,1,0] += np.imag(X5)
					outputs['Im_wave_forces'][i,2,0] += np.imag(X7)