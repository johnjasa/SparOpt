import numpy as np

from openmdao.api import ExplicitComponent

class NormFairleadMWind(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)


		self.add_input('Re_RAO_Mwind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_Mwind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_Mwind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_Mwind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('Re_RAO_Mwind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_output('Im_RAO_Mwind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')

		self.declare_partials('Re_RAO_Mwind_fairlead', 'Re_RAO_Mwind_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_fairlead', 'Re_RAO_Mwind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_fairlead', 'z_moor')
		self.declare_partials('Im_RAO_Mwind_fairlead', 'Im_RAO_Mwind_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_fairlead', 'Im_RAO_Mwind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_fairlead', 'z_moor')

	def compute(self, inputs, outputs):
		RAO_Mwind_surge = inputs['Re_RAO_Mwind_surge'] + 1j * inputs['Im_RAO_Mwind_surge']
		RAO_Mwind_pitch = inputs['Re_RAO_Mwind_pitch'] + 1j * inputs['Im_RAO_Mwind_pitch']
		z_moor = inputs['z_moor']

		RAO_Mwind_fairlead = RAO_Mwind_surge + z_moor * RAO_Mwind_pitch

		outputs['Re_RAO_Mwind_fairlead'] = np.real(RAO_Mwind_fairlead)

		outputs['Im_RAO_Mwind_fairlead'] = np.imag(RAO_Mwind_fairlead)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		z_moor = inputs['z_moor']

		RAO_Mwind_surge = inputs['Re_RAO_Mwind_surge'] + 1j * inputs['Im_RAO_Mwind_surge']
		RAO_Mwind_pitch = inputs['Re_RAO_Mwind_pitch'] + 1j * inputs['Im_RAO_Mwind_pitch']
		
		partials['Re_RAO_Mwind_fairlead', 'Re_RAO_Mwind_surge'] = np.ones(N_omega)
		partials['Re_RAO_Mwind_fairlead', 'Re_RAO_Mwind_pitch'] = np.ones(N_omega) * z_moor
		partials['Re_RAO_Mwind_fairlead', 'z_moor'] = np.real(RAO_Mwind_pitch)

		partials['Im_RAO_Mwind_fairlead', 'Im_RAO_Mwind_surge'] = np.ones(N_omega)
		partials['Im_RAO_Mwind_fairlead', 'Im_RAO_Mwind_pitch'] = np.ones(N_omega) * z_moor
		partials['Im_RAO_Mwind_fairlead', 'z_moor'] = np.imag(RAO_Mwind_pitch)