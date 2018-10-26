import numpy as np
import scipy.special as ss
import scipy.optimize as so

from openmdao.api import ExplicitComponent

class WaveLoads(ExplicitComponent):

	def setup(self):
		self.add_input('x_sparelem', val=np.zeros(13), units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('omega_wave', val=np.zeros(80), units='rad/s')
		self.add_input('water_depth', val=0., units='m')

		self.add_output('Re_wave_forces', val=np.zeros((80,3,1)))
		self.add_output('Im_wave_forces', val=np.zeros((80,3,1)))

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		Z_spar = inputs['Z_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		omega_wave = inputs['omega_wave']
		h = inputs['water_depth'][0]

		outputs['Re_wave_forces'] = np.zeros((80,3,1))
		outputs['Im_wave_forces'] = np.zeros((80,3,1))

		N_elem = len(x_sparelem)

		a = 0.

		wavenum = np.zeros(80)

		def F(x): #define implicit equation for wave number
			return x*9.80665*np.tanh(x*h) - omega_wave[i]**2.

		for i in xrange(80):
			wavenum = so.broyden1(F,1.01*omega_wave[i]**2./9.80665)
			if wavenum < 0.0:
				wavenum = -wavenum
			if (wavenum * h) > 710: #upper limit for numpy cosh function (returns inf for larger numbers)
				continue

			for j in xrange(N_elem):
				z = (z_sparnode[j] + z_sparnode[j+1]) / 2
				dz = z_sparnode[j+1] - z_sparnode[j]

				if z <= 0.:
					for k in xrange(len(Z_spar) - 1):
						if (z < Z_spar[k+1]) and (z >= Z_spar[k]):
							a = D_spar[k] / 2.
							break

					J = ss.jvp(1,wavenum*a,1)
					Y = ss.yvp(1,wavenum*a,1)
					G = 1. / np.sqrt(J**2. + Y**2.)
					alpha = np.arctan2(J,Y)
				
					X1 = 4. * 1025. * 9.80665 / wavenum * np.cosh(wavenum * (z + h)) / np.cosh(wavenum * h) * G * dz * np.exp(1j * alpha + 1j * np.pi / 2.)
					X5 = 4. * 1025. * 9.80665 / wavenum * np.cosh(wavenum * (z + h)) / np.cosh(wavenum * h) * G * dz * z * np.exp(1j * alpha + 1j * np.pi / 2.)
					X7 = 4. * 1025. * 9.80665 / wavenum * np.cosh(wavenum * (z + h)) / np.cosh(wavenum * h) * G * dz * x_sparelem[j] * np.exp(1j * alpha + 1j * np.pi / 2.)

					outputs['Re_wave_forces'][i,0,0] += np.real(X1)
					outputs['Re_wave_forces'][i,1,0] += np.real(X5)
					outputs['Re_wave_forces'][i,2,0] += np.real(X7)

					outputs['Im_wave_forces'][i,0,0] += np.imag(X1)
					outputs['Im_wave_forces'][i,1,0] += np.imag(X5)
					outputs['Im_wave_forces'][i,2,0] += np.imag(X7)