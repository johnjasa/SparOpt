import numpy as np

from openmdao.api import ExplicitComponent

class NormFairleadWave(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)


		self.add_input('Re_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('Re_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_output('Im_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')

		self.declare_partials('Re_RAO_wave_fairlead', 'Re_RAO_wave_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_fairlead', 'Re_RAO_wave_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_fairlead', 'z_moor')
		self.declare_partials('Im_RAO_wave_fairlead', 'Im_RAO_wave_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_fairlead', 'Im_RAO_wave_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_fairlead', 'z_moor')

	def compute(self, inputs, outputs):
		RAO_wave_surge = inputs['Re_RAO_wave_surge'] + 1j * inputs['Im_RAO_wave_surge']
		RAO_wave_pitch = inputs['Re_RAO_wave_pitch'] + 1j * inputs['Im_RAO_wave_pitch']
		z_moor = inputs['z_moor']

		RAO_wave_fairlead = RAO_wave_surge + z_moor * RAO_wave_pitch

		outputs['Re_RAO_wave_fairlead'] = np.real(RAO_wave_fairlead)

		outputs['Im_RAO_wave_fairlead'] = np.imag(RAO_wave_fairlead)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		z_moor = inputs['z_moor']

		RAO_wave_surge = inputs['Re_RAO_wave_surge'] + 1j * inputs['Im_RAO_wave_surge']
		RAO_wave_pitch = inputs['Re_RAO_wave_pitch'] + 1j * inputs['Im_RAO_wave_pitch']
		
		partials['Re_RAO_wave_fairlead', 'Re_RAO_wave_surge'] = np.ones(N_omega)
		partials['Re_RAO_wave_fairlead', 'Re_RAO_wave_pitch'] = np.ones(N_omega) * z_moor
		partials['Re_RAO_wave_fairlead', 'z_moor'] = np.real(RAO_wave_pitch)

		partials['Im_RAO_wave_fairlead', 'Im_RAO_wave_surge'] = np.ones(N_omega)
		partials['Im_RAO_wave_fairlead', 'Im_RAO_wave_pitch'] = np.ones(N_omega) * z_moor
		partials['Im_RAO_wave_fairlead', 'z_moor'] = np.imag(RAO_wave_pitch)