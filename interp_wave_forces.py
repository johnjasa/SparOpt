import numpy as np

from openmdao.api import ExplicitComponent

class InterpWaveForces(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		self.omega_wave = freqs['omega_wave']
		N_omega = len(self.omega)
		N_omega_wave = len(self.omega_wave)

		self.add_input('Re_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')
		self.add_input('Im_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')

		self.add_output('Re_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_output('Im_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_output('Re_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_output('Im_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_output('Re_wave_force_bend', val=np.zeros(N_omega), units='N/m')
		self.add_output('Im_wave_force_bend', val=np.zeros(N_omega), units='N/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega
		omega_wave = self.omega_wave

		wave_forces = inputs['Re_wave_forces'] + 1j * inputs['Im_wave_forces']
		wave_force_surge = wave_forces[:,0,0]
		wave_force_pitch = wave_forces[:,1,0]
		wave_force_bend = wave_forces[:,2,0]

		wave_force_surge = np.interp(omega, omega_wave, wave_force_surge)
		wave_force_pitch = np.interp(omega, omega_wave, wave_force_pitch)
		wave_force_bend = np.interp(omega, omega_wave, wave_force_bend)

		outputs['Re_wave_force_surge'] = np.real(wave_force_surge)
		outputs['Re_wave_force_pitch'] = np.real(wave_force_pitch)
		outputs['Re_wave_force_bend'] = np.real(wave_force_bend)

		outputs['Im_wave_force_surge'] = np.imag(wave_force_surge)
		outputs['Im_wave_force_pitch'] = np.imag(wave_force_pitch)
		outputs['Im_wave_force_bend'] = np.imag(wave_force_bend)
"""
	def compute_partials(self, inputs, partials):
		omega = self.omega
		omega_wave = self.omega_wave
		N_omega = len(omega)
		N_omega_wave = len(omega_wave)

		partials['Re_wave_force_surge', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Im_wave_force_surge', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Re_wave_force_pitch', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Im_wave_force_pitch', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Re_wave_force_bend', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Im_wave_force_bend', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))

		partials['Re_wave_force_surge', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Im_wave_force_surge', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Re_wave_force_pitch', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Im_wave_force_pitch', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Re_wave_force_bend', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
		partials['Im_wave_force_bend', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
"""