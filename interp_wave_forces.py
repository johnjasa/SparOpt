import numpy as np
from scipy.spatial.kdtree import KDTree

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
		domega_wave = self.omega_wave[1] - self.omega_wave[0]

		self.add_input('Re_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')
		self.add_input('Im_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')

		self.add_output('Re_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_output('Im_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_output('Re_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_output('Im_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_output('Re_wave_force_bend', val=np.zeros(N_omega), units='N/m')
		self.add_output('Im_wave_force_bend', val=np.zeros(N_omega), units='N/m')

		omega_wave_t = np.array([self.omega_wave]).T
		tree = KDTree(omega_wave_t)

		self.surge_cols = []
		self.pitch_cols = []
		self.bend_cols = []

		for i in xrange(len(self.omega)):
			dist, idx = KDTree.query(tree,np.array([self.omega[i]]), k=2)

			if dist[0] > domega_wave or dist[1] > domega_wave:
				self.surge_cols.append(0)
				self.surge_cols.append(0)
				self.pitch_cols.append(0)
				self.pitch_cols.append(0)
				self.bend_cols.append(0)
				self.bend_cols.append(0)
			else:
				idx0 = np.min(idx)
				idx1 = np.max(idx)

				self.surge_cols.append(3*idx0)
				self.surge_cols.append(3*idx1)
				self.pitch_cols.append(3*idx0+1)
				self.pitch_cols.append(3*idx1+1)
				self.bend_cols.append(3*idx0+2)
				self.bend_cols.append(3*idx1+2)

		self.declare_partials('Re_wave_force_surge', 'Re_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.surge_cols)
		self.declare_partials('Re_wave_force_surge', 'Im_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.surge_cols)
		self.declare_partials('Im_wave_force_surge', 'Re_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.surge_cols)
		self.declare_partials('Im_wave_force_surge', 'Im_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.surge_cols)
		self.declare_partials('Re_wave_force_pitch', 'Re_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.pitch_cols)
		self.declare_partials('Re_wave_force_pitch', 'Im_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.pitch_cols)
		self.declare_partials('Im_wave_force_pitch', 'Re_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.pitch_cols)
		self.declare_partials('Im_wave_force_pitch', 'Im_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.pitch_cols)
		self.declare_partials('Re_wave_force_bend', 'Re_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.bend_cols)
		self.declare_partials('Re_wave_force_bend', 'Im_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.bend_cols)
		self.declare_partials('Im_wave_force_bend', 'Re_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.bend_cols)
		self.declare_partials('Im_wave_force_bend', 'Im_wave_forces', method='cs') #rows=np.repeat(np.arange(N_omega),2), cols=self.bend_cols)

	def compute(self, inputs, outputs):
		omega = self.omega
		omega_wave = self.omega_wave

		wave_forces = inputs['Re_wave_forces'] + 1j * inputs['Im_wave_forces']
		wave_force_surge = wave_forces[:,0,0]
		wave_force_pitch = wave_forces[:,1,0]
		wave_force_bend = wave_forces[:,2,0]

		wave_force_surge = np.interp(omega, omega_wave, wave_force_surge, left=0., right=0.)
		wave_force_pitch = np.interp(omega, omega_wave, wave_force_pitch, left=0., right=0.)
		wave_force_bend = np.interp(omega, omega_wave, wave_force_bend, left=0., right=0.)

		outputs['Re_wave_force_surge'] = np.real(wave_force_surge)
		outputs['Re_wave_force_pitch'] = np.real(wave_force_pitch)
		outputs['Re_wave_force_bend'] = np.real(wave_force_bend)

		outputs['Im_wave_force_surge'] = np.imag(wave_force_surge)
		outputs['Im_wave_force_pitch'] = np.imag(wave_force_pitch)
		outputs['Im_wave_force_bend'] = np.imag(wave_force_bend)

	# def compute_partials(self, inputs, partials):
	# 	omega = self.omega
	# 	omega_wave = self.omega_wave
	# 	N_omega = len(omega)
	# 	N_omega_wave = len(omega_wave)
	# 	domega_wave = omega_wave[1] - omega_wave[0]
	#
	# 	surge_cols = self.surge_cols
	# 	pitch_cols = self.pitch_cols
	# 	bend_cols = self.bend_cols
	#
	# 	for i in xrange(N_omega):
	# 		if not surge_cols[2*i+1] == 0:
	# 			partials['Re_wave_force_surge', 'Re_wave_forces'][2*i] = (omega_wave[surge_cols[2*i+1]/3] - omega[i]) / domega_wave
	# 			partials['Re_wave_force_surge', 'Re_wave_forces'][2*i+1] = (omega[i] - omega_wave[surge_cols[2*i]/3]) / domega_wave
	# 			partials['Im_wave_force_surge', 'Im_wave_forces'][2*i] = (omega_wave[surge_cols[2*i+1]/3] - omega[i]) / domega_wave
	# 			partials['Im_wave_force_surge', 'Im_wave_forces'][2*i+1] = (omega[i] - omega_wave[surge_cols[2*i]/3]) / domega_wave
	# 			partials['Re_wave_force_pitch', 'Re_wave_forces'][2*i] = (omega_wave[pitch_cols[2*i+1]/3] - omega[i]) / domega_wave
	# 			partials['Re_wave_force_pitch', 'Re_wave_forces'][2*i+1] = (omega[i] - omega_wave[pitch_cols[2*i]/3]) / domega_wave
	# 			partials['Im_wave_force_pitch', 'Im_wave_forces'][2*i] = (omega_wave[pitch_cols[2*i+1]/3] - omega[i]) / domega_wave
	# 			partials['Im_wave_force_pitch', 'Im_wave_forces'][2*i+1] = (omega[i] - omega_wave[pitch_cols[2*i]/3]) / domega_wave
	# 			partials['Re_wave_force_bend', 'Re_wave_forces'][2*i] = (omega_wave[bend_cols[2*i+1]/3] - omega[i]) / domega_wave
	# 			partials['Re_wave_force_bend', 'Re_wave_forces'][2*i+1] = (omega[i] - omega_wave[bend_cols[2*i]/3]) / domega_wave
	# 			partials['Im_wave_force_bend', 'Im_wave_forces'][2*i] = (omega_wave[bend_cols[2*i+1]/3] - omega[i]) / domega_wave
	# 			partials['Im_wave_force_bend', 'Im_wave_forces'][2*i+1] = (omega[i] - omega_wave[bend_cols[2*i]/3]) / domega_wave
	#
	# 	"""
	# 	partials['Re_wave_force_surge', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Im_wave_force_surge', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Re_wave_force_pitch', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Im_wave_force_pitch', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Re_wave_force_bend', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Im_wave_force_bend', 'Re_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	#
	# 	partials['Re_wave_force_surge', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Im_wave_force_surge', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Re_wave_force_pitch', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Im_wave_force_pitch', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Re_wave_force_bend', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	partials['Im_wave_force_bend', 'Im_wave_forces'] = np.zeros((N_omega,N_omega_wave*3))
	# 	"""
