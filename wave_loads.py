import numpy as np
import scipy.special as ss

from openmdao.api import ExplicitComponent

class WaveLoads(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega_wave = freqs['omega_wave']
		N_omega_wave = len(self.omega_wave)

		self.add_input('x_sparelem', val=np.zeros(13), units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('wave_number', val=np.zeros(N_omega_wave), units='1/m')
		self.add_input('water_depth', val=0., units='m')

		self.add_output('Re_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')
		self.add_output('Im_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega_wave = self.omega_wave
		N_omega_wave = len(omega_wave)

		D_spar = inputs['D_spar']
		Z_spar = inputs['Z_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		wave_number = inputs['wave_number']
		h = inputs['water_depth'][0]
		outputs['Re_wave_forces'] = np.zeros((N_omega_wave,3,1))
		outputs['Im_wave_forces'] = np.zeros((N_omega_wave,3,1))

		N_elem = len(x_sparelem)

		a = 0.

		for i in xrange(N_omega_wave):
			if (wave_number[i] * h) >= 700: #upper limit for numpy cosh function (returns inf for larger numbers)
				continue

			for j in xrange(N_elem):
				z = (z_sparnode[j] + z_sparnode[j+1]) / 2
				dz = z_sparnode[j+1] - z_sparnode[j]

				if z <= 0.:
					for k in xrange(len(Z_spar) - 1):
						if (z < Z_spar[k+1]) and (z >= Z_spar[k]):
							a = D_spar[k] / 2.
							break

					J = ss.jvp(1,wave_number[i]*a,1)
					Y = ss.yvp(1,wave_number[i]*a,1)
					G = 1. / np.sqrt(J**2. + Y**2.)
					alpha = np.arctan2(J,Y)
				
					X1 = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * np.exp(1j * alpha + 1j * np.pi / 2.)
					X5 = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * z * np.exp(1j * alpha + 1j * np.pi / 2.)
					X7 = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * x_sparelem[j] * np.exp(1j * alpha + 1j * np.pi / 2.)

					outputs['Re_wave_forces'][i,0,0] += np.real(X1)
					outputs['Re_wave_forces'][i,1,0] += np.real(X5)
					outputs['Re_wave_forces'][i,2,0] += np.real(X7)

					outputs['Im_wave_forces'][i,0,0] += np.imag(X1)
					outputs['Im_wave_forces'][i,1,0] += np.imag(X5)
					outputs['Im_wave_forces'][i,2,0] += np.imag(X7)

	def compute_partials(self, inputs, partials):
		omega_wave = self.omega_wave
		N_omega_wave = len(omega_wave)

		D_spar = inputs['D_spar']
		Z_spar = inputs['Z_spar']
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']
		wave_number = inputs['wave_number']
		h = inputs['water_depth'][0]

		dwave_forces_dx_sparelem = np.zeros((3 * N_omega_wave,13), dtype=complex)
		dwave_forces_dz_sparnode = np.zeros((3 * N_omega_wave,14), dtype=complex)
		dwave_forces_dD_spar = np.zeros((3 * N_omega_wave,10), dtype=complex)
		dwave_forces_dwave_number = np.zeros((3 * N_omega_wave,N_omega_wave), dtype=complex)
		dwave_forces_dwater_depth = np.zeros((3 * N_omega_wave,1), dtype=complex)

		N_elem = len(x_sparelem)

		a = 0.

		for i in xrange(N_omega_wave):
			if (wave_number[i] * h) >= 700: #upper limit for numpy cosh function (returns inf for larger numbers)
				continue
			for j in xrange(N_elem):
				z = (z_sparnode[j] + z_sparnode[j+1]) / 2
				dz = z_sparnode[j+1] - z_sparnode[j]

				if z <= 0.:
					for k in xrange(len(Z_spar) - 1):
						if (z < Z_spar[k+1]) and (z >= Z_spar[k]):
							a = D_spar[k] / 2.
							break

					J = ss.jvp(1,wave_number[i]*a,1)
					Y = ss.yvp(1,wave_number[i]*a,1)
					G = 1. / np.sqrt(J**2. + Y**2.)
					alpha = np.arctan2(J,Y)
					Jd = ss.jvp(1,wave_number[i]*a,2)
					Yd = ss.yvp(1,wave_number[i]*a,2)
				
					X1 = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * np.exp(1j * alpha + 1j * np.pi / 2.)
					X5 = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * z * np.exp(1j * alpha + 1j * np.pi / 2.)
					X7 = 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * x_sparelem[j] * np.exp(1j * alpha + 1j * np.pi / 2.)

					dwave_forces_dD_spar[3*i,k] += 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * dz * np.exp(1j * np.pi / 2.) * (-0.5 * (J**2. + Y**2.)**(-1.5) * wave_number[i] * (J * Jd + Y * Yd) * np.exp(1j * alpha) + G * np.exp(1j * alpha) * 1j * 1. / (1. + (J / Y)**2.) * wave_number[i] / 2. * (Jd / Y - J / Y**2. * Yd))
					dwave_forces_dD_spar[3*i+1,k] += 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * dz * z * np.exp(1j * np.pi / 2.) * (-0.5 * (J**2. + Y**2.)**(-1.5) * wave_number[i] * (J * Jd + Y * Yd) * np.exp(1j * alpha) + G * np.exp(1j * alpha) * 1j * 1. / (1. + (J / Y)**2.) * wave_number[i] / 2. * (Jd / Y - J / Y**2. * Yd))
					dwave_forces_dD_spar[3*i+2,k] += 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * dz * x_sparelem[j] * np.exp(1j * np.pi / 2.) * (-0.5 * (J**2. + Y**2.)**(-1.5) * wave_number[i] * (J * Jd + Y * Yd) * np.exp(1j * alpha) + G * np.exp(1j * alpha) * 1j * 1. / (1. + (J / Y)**2.) * wave_number[i] / 2. * (Jd / Y - J / Y**2. * Yd))

					dwave_forces_dx_sparelem[3*i+2,j] += 4. * 1025. * 9.80665 / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * dz * np.exp(1j * alpha + 1j * np.pi / 2.)

					dwave_forces_dz_sparnode[3*i,j] += 4. * 1025. * 9.80665 / wave_number[i] * 1. / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha + 1j * np.pi / 2.) * (-np.cosh(wave_number[i] * (z + h)) + dz * wave_number[i] * 0.5 * np.sinh(wave_number[i] * (z + h)))
					dwave_forces_dz_sparnode[3*i,j+1] += 4. * 1025. * 9.80665 / wave_number[i] * 1. / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha + 1j * np.pi / 2.) * (np.cosh(wave_number[i] * (z + h)) + dz * wave_number[i] * 0.5 * np.sinh(wave_number[i] * (z + h)))
					dwave_forces_dz_sparnode[3*i+1,j] += 4. * 1025. * 9.80665 / wave_number[i] * 1. / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha + 1j * np.pi / 2.) * (-np.cosh(wave_number[i] * (z + h)) * z + dz * z * wave_number[i] * 0.5 * np.sinh(wave_number[i] * (z + h)) + np.cosh(wave_number[i] * (z + h)) * dz * 0.5)
					dwave_forces_dz_sparnode[3*i+1,j+1] += 4. * 1025. * 9.80665 / wave_number[i] * 1. / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha + 1j * np.pi / 2.) * (np.cosh(wave_number[i] * (z + h)) * z + dz * z * wave_number[i] * 0.5 * np.sinh(wave_number[i] * (z + h)) + np.cosh(wave_number[i] * (z + h)) * dz * 0.5)
					dwave_forces_dz_sparnode[3*i+2,j] += 4. * 1025. * 9.80665 / wave_number[i] * 1. / np.cosh(wave_number[i] * h) * G * x_sparelem[j] * np.exp(1j * alpha + 1j * np.pi / 2.) * (-np.cosh(wave_number[i] * (z + h)) + dz * wave_number[i] * 0.5 * np.sinh(wave_number[i] * (z + h)))
					dwave_forces_dz_sparnode[3*i+2,j+1] += 4. * 1025. * 9.80665 / wave_number[i] * 1. / np.cosh(wave_number[i] * h) * G * x_sparelem[j] * np.exp(1j * alpha + 1j * np.pi / 2.) * (np.cosh(wave_number[i] * (z + h)) + dz * wave_number[i] * 0.5 * np.sinh(wave_number[i] * (z + h)))

					dwave_forces_dwave_number[3*i,i] += 4. * 1025. * 9.80665 * dz * np.exp(1j * np.pi / 2.) * (-1. / wave_number[i]**2. * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) + 1. / wave_number[i] * (z + h) * np.sinh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) - 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) * np.tanh(wave_number[i] * h) * 1. / np.cosh(wave_number[i] * h) * h * G * np.exp(1j * alpha) + 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * (-0.5) * (J**2. + Y**2.)**(-1.5) * a * (2. * J * Jd + 2. * Y * Yd) * np.exp(1j * alpha) + 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) * 1j * 1. / (1. + (J / Y)**2.) * a * (Jd / Y - J / Y**2. * Yd))
					dwave_forces_dwave_number[3*i+1,i] += 4. * 1025. * 9.80665 * dz * z * np.exp(1j * np.pi / 2.) * (-1. / wave_number[i]**2. * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) + 1. / wave_number[i] * (z + h) * np.sinh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) - 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h))  * np.tanh(wave_number[i] * h) * 1. / np.cosh(wave_number[i] * h) * h * G * np.exp(1j * alpha) + 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * (-0.5) * (J**2. + Y**2.)**(-1.5) * a * (2. * J * Jd + 2. * Y * Yd) * np.exp(1j * alpha) + 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) * 1j * 1. / (1. + (J / Y)**2.) * a * (Jd / Y - J / Y**2. * Yd))
					dwave_forces_dwave_number[3*i+2,i] += 4. * 1025. * 9.80665 * dz * x_sparelem[j] * np.exp(1j * np.pi / 2.) * (-1. / wave_number[i]**2. * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) + 1. / wave_number[i] * (z + h) * np.sinh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) - 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h))  * np.tanh(wave_number[i] * h) * 1. / np.cosh(wave_number[i] * h) * h * G * np.exp(1j * alpha) + 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * (-0.5) * (J**2. + Y**2.)**(-1.5) * a * (2. * J * Jd + 2. * Y * Yd) * np.exp(1j * alpha) + 1. / wave_number[i] * np.cosh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) * G * np.exp(1j * alpha) * 1j * 1. / (1. + (J / Y)**2.) * a * (Jd / Y - J / Y**2. * Yd))

					dwave_forces_dwater_depth[3*i,0] += 4. * 1025. * 9.80665 / wave_number[i] * G * dz * np.exp(1j * alpha + 1j * np.pi / 2.) * (wave_number[i] * np.sinh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) - wave_number[i] * np.cosh(wave_number[i] * (z + h)) * np.tanh(wave_number[i] * h) * 1. / np.cosh(wave_number[i] * h))
					dwave_forces_dwater_depth[3*i+1,0] += 4. * 1025. * 9.80665 / wave_number[i] * G * dz * z * np.exp(1j * alpha + 1j * np.pi / 2.) * (wave_number[i] * np.sinh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) - wave_number[i] * np.cosh(wave_number[i] * (z + h)) * np.tanh(wave_number[i] * h) * 1. / np.cosh(wave_number[i] * h))
					dwave_forces_dwater_depth[3*i+2,0] += 4. * 1025. * 9.80665 / wave_number[i] * G * dz * x_sparelem[j] * np.exp(1j * alpha + 1j * np.pi / 2.) * (wave_number[i] * np.sinh(wave_number[i] * (z + h)) / np.cosh(wave_number[i] * h) - wave_number[i] * np.cosh(wave_number[i] * (z + h)) * np.tanh(wave_number[i] * h) * 1. / np.cosh(wave_number[i] * h))

		partials['Re_wave_forces', 'D_spar'] = np.real(dwave_forces_dD_spar)
		partials['Re_wave_forces', 'D_spar'] = np.real(dwave_forces_dD_spar)
		partials['Re_wave_forces', 'D_spar'] = np.real(dwave_forces_dD_spar)

		partials['Re_wave_forces', 'x_sparelem'] = np.real(dwave_forces_dx_sparelem)

		partials['Re_wave_forces', 'z_sparnode'] = np.real(dwave_forces_dz_sparnode)
		partials['Re_wave_forces', 'z_sparnode'] = np.real(dwave_forces_dz_sparnode)
		partials['Re_wave_forces', 'z_sparnode'] = np.real(dwave_forces_dz_sparnode)
		partials['Re_wave_forces', 'z_sparnode'] = np.real(dwave_forces_dz_sparnode)
		partials['Re_wave_forces', 'z_sparnode'] = np.real(dwave_forces_dz_sparnode)
		partials['Re_wave_forces', 'z_sparnode'] = np.real(dwave_forces_dz_sparnode)

		partials['Re_wave_forces', 'wave_number'] = np.real(dwave_forces_dwave_number)
		partials['Re_wave_forces', 'wave_number'] = np.real(dwave_forces_dwave_number)
		partials['Re_wave_forces', 'wave_number'] = np.real(dwave_forces_dwave_number)

		partials['Re_wave_forces', 'water_depth'] = np.real(dwave_forces_dwater_depth)
		partials['Re_wave_forces', 'water_depth'] = np.real(dwave_forces_dwater_depth)
		partials['Re_wave_forces', 'water_depth'] = np.real(dwave_forces_dwater_depth)

		partials['Im_wave_forces', 'D_spar'] = np.imag(dwave_forces_dD_spar)
		partials['Im_wave_forces', 'D_spar'] = np.imag(dwave_forces_dD_spar)
		partials['Im_wave_forces', 'D_spar'] = np.imag(dwave_forces_dD_spar)

		partials['Im_wave_forces', 'x_sparelem'] = np.imag(dwave_forces_dx_sparelem)

		partials['Im_wave_forces', 'z_sparnode'] = np.imag(dwave_forces_dz_sparnode)
		partials['Im_wave_forces', 'z_sparnode'] = np.imag(dwave_forces_dz_sparnode)
		partials['Im_wave_forces', 'z_sparnode'] = np.imag(dwave_forces_dz_sparnode)
		partials['Im_wave_forces', 'z_sparnode'] = np.imag(dwave_forces_dz_sparnode)
		partials['Im_wave_forces', 'z_sparnode'] = np.imag(dwave_forces_dz_sparnode)
		partials['Im_wave_forces', 'z_sparnode'] = np.imag(dwave_forces_dz_sparnode)

		partials['Im_wave_forces', 'wave_number'] = np.imag(dwave_forces_dwave_number)
		partials['Im_wave_forces', 'wave_number'] = np.imag(dwave_forces_dwave_number)
		partials['Im_wave_forces', 'wave_number'] = np.imag(dwave_forces_dwave_number)

		partials['Im_wave_forces', 'water_depth'] = np.imag(dwave_forces_dwater_depth)
		partials['Im_wave_forces', 'water_depth'] = np.imag(dwave_forces_dwater_depth)
		partials['Im_wave_forces', 'water_depth'] = np.imag(dwave_forces_dwater_depth)