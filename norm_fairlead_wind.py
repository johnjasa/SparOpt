import numpy as np

from openmdao.api import ExplicitComponent

class NormFairleadWind(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)


		self.add_input('Re_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('Re_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_output('Im_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')

		self.declare_partials('Re_RAO_wind_fairlead', 'Re_RAO_wind_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_fairlead', 'Re_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_fairlead', 'z_moor')
		self.declare_partials('Im_RAO_wind_fairlead', 'Im_RAO_wind_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_fairlead', 'Im_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_fairlead', 'z_moor')

	def compute(self, inputs, outputs):
		RAO_wind_surge = inputs['Re_RAO_wind_surge'] + 1j * inputs['Im_RAO_wind_surge']
		RAO_wind_pitch = inputs['Re_RAO_wind_pitch'] + 1j * inputs['Im_RAO_wind_pitch']
		z_moor = inputs['z_moor']

		RAO_wind_fairlead = RAO_wind_surge + z_moor * RAO_wind_pitch

		outputs['Re_RAO_wind_fairlead'] = np.real(RAO_wind_fairlead)

		outputs['Im_RAO_wind_fairlead'] = np.imag(RAO_wind_fairlead)

	def compute_partials(self, inputs, partials): #TODO check
		omega = self.omega
		N_omega = len(omega)

		z_moor = inputs['z_moor']

		RAO_wind_surge = inputs['Re_RAO_wind_surge'] + 1j * inputs['Im_RAO_wind_surge']
		RAO_wind_pitch = inputs['Re_RAO_wind_pitch'] + 1j * inputs['Im_RAO_wind_pitch']
		
		partials['Re_RAO_wind_fairlead', 'Re_RAO_wind_surge'] = np.ones(N_omega)
		partials['Re_RAO_wind_fairlead', 'Re_RAO_wind_pitch'] = np.ones(N_omega) * z_moor
		partials['Re_RAO_wind_fairlead', 'z_moor'] = np.real(RAO_wind_pitch)

		partials['Im_RAO_wind_fairlead', 'Im_RAO_wind_surge'] = np.ones(N_omega)
		partials['Im_RAO_wind_fairlead', 'Im_RAO_wind_pitch'] = np.ones(N_omega) * z_moor
		partials['Im_RAO_wind_fairlead', 'z_moor'] = np.imag(RAO_wind_pitch)