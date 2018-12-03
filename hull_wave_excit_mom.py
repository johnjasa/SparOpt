import numpy as np
import scipy.special as ss

from openmdao.api import ExplicitComponent

class HullWaveExcitMom(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega_wave = freqs['omega_wave']
		N_omega_wave = len(self.omega_wave)

		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('wave_number', val=np.zeros(N_omega_wave), units='1/m')
		self.add_input('water_depth', val=0., units='m')

		self.add_output('Re_hull_wave_mom', val=np.zeros((N_omega_wave,10)), units='N*m/m')
		self.add_output('Im_hull_wave_mom', val=np.zeros((N_omega_wave,10)), units='N*m/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega_wave = self.omega_wave
		N_omega_wave = len(omega_wave)

		D_spar = inputs['D_spar']
		Z_spar = inputs['Z_spar']
		wave_number = inputs['wave_number']
		h = inputs['water_depth'][0]
		outputs['Re_hull_wave_mom'] = np.zeros((N_omega_wave,10))
		outputs['Im_hull_wave_mom'] = np.zeros((N_omega_wave,10))

		N_elem = len(D_spar)

		a = 0.

		for i in xrange(N_omega_wave):
			if (wave_number[i] * h) >= 700: #upper limit for numpy cosh function (returns inf for larger numbers)
				continue

			for j in xrange(N_elem):
				if j == (N_elem - 1): #surface-piercing element
					z = (Z_spar[j] + 0.) / 2
					dz = 0. - Z_spar[j]
				else:
					z = (Z_spar[j] + Z_spar[j+1]) / 2
					dz = Z_spar[j+1] - Z_spar[j]

				a = D_spar[j] / 2.

				J = ss.jvp(1,wave_number[i]*a,1)
				Y = ss.yvp(1,wave_number[i]*a,1)
				G = 1. / np.sqrt(J**2. + Y**2.)
				alpha = np.arctan2(J,Y)
				
				Fx = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * np.exp(1j * alpha + 1j * np.pi / 2.)

				for k in xrange(j):
					outputs['Re_hull_wave_mom'][i,k] += np.real(Fx) * (z - Z_spar[k])
					outputs['Im_hull_wave_mom'][i,k] += np.imag(Fx) * (z - Z_spar[k])