import numpy as np
import scipy.optimize as so

from openmdao.api import ImplicitComponent

class WaveNumber(ImplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega_wave = freqs['omega_wave']
		N_omega_wave = len(self.omega_wave)

		self.add_input('water_depth', val=0., units='m')

		self.add_output('wave_number', val=np.ones(N_omega_wave)*0.05, units='1/m')

		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		omega_wave = self.omega_wave
		N_omega_wave = len(omega_wave)

		h = inputs['water_depth'][0]

		residuals['wave_number'] = np.zeros(N_omega_wave)

		for i in xrange(N_omega_wave):
			residuals['wave_number'][i] = outputs['wave_number'][i] * 9.80665 * np.tanh(outputs['wave_number'][i] * h) - omega_wave[i]**2.

	def solve_nonlinear(self, inputs, outputs):
		omega_wave = self.omega_wave
		N_omega_wave = len(omega_wave)

		h = inputs['water_depth'][0]

		outputs['wave_number'] = np.zeros(N_omega_wave)

		def F(x):
			return x * 9.80665 * np.tanh(x * h) - omega_wave[i]**2.

		for i in xrange(N_omega_wave):
			wavenum = so.broyden1(F, 1.1 * omega_wave[i]**2. / 9.80665)
			if wavenum < 0.0:
				wavenum = -wavenum

			outputs['wave_number'][i] = wavenum

	def linearize(self, inputs, outputs, partials):
		omega_wave = self.omega_wave
		N_omega_wave = len(omega_wave)

		h = inputs['water_depth'][0]

		partials['wave_number', 'water_depth'] = np.zeros((N_omega_wave,1))
		partials['wave_number', 'wave_number'] = np.zeros((N_omega_wave,N_omega_wave))

		for i in xrange(N_omega_wave):
			partials['wave_number', 'water_depth'][i,0] = outputs['wave_number'][i]**2. * 9.80665 * (1. - np.tanh(outputs['wave_number'][i] * h)**2.)
			partials['wave_number', 'wave_number'][i,i] = 9.80665 * np.tanh(outputs['wave_number'][i] * h) + outputs['wave_number'][i] * 9.80665 * h * (1. - np.tanh(outputs['wave_number'][i] * h)**2.)