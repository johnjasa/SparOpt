import numpy as np

from openmdao.api import ExplicitComponent

class NormMoorTenDynWind(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)


		self.add_input('Re_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('phi_upper_end', val=0., units='rad')
		self.add_input('k_e_moor', val=0., units='N/m')
		self.add_input('k_g_moor', val=0., units='N/m')
		self.add_input('gen_m_moor', val=0., units='kg')
		self.add_input('gen_c_moor', val=0., units='N*s/m')

		self.add_output('Re_RAO_wind_moor_ten_dyn', val=np.zeros(N_omega), units='N/(m/s)')
		self.add_output('Im_RAO_wind_moor_ten_dyn', val=np.zeros(N_omega), units='N/(m/s)')

		self.declare_partials('Re_RAO_wind_moor_ten_dyn', 'Re_RAO_wind_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_moor_ten_dyn', 'phi_upper_end')
		self.declare_partials('Im_RAO_wind_moor_ten_dyn', 'Im_RAO_wind_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_moor_ten_dyn', 'phi_upper_end')

	def compute(self, inputs, outputs):
		omega = self.omega

		RAO_wind_fairlead = inputs['Re_RAO_wind_fairlead'] + 1j * inputs['Im_RAO_wind_fairlead']
		phi = inputs['phi_upper_end']
		k_e = inputs['k_e_moor']
		k_g = inputs['k_g_moor']
		m = inputs['gen_m_moor']
		c = inputs['gen_c_moor']

		RAO_wind_moor_ten = np.cos(phi) * k_e * (-omega**2. * m + 1j * omega * c + k_g) / (-omega**2. * m + 1j * omega * c + k_g + k_e) * RAO_wind_fairlead

		outputs['Re_RAO_wind_moor_ten_dyn'] = np.real(RAO_wind_moor_ten)

		outputs['Im_RAO_wind_moor_ten_dyn'] = np.imag(RAO_wind_moor_ten)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		K_moor = inputs['K_moor']

		RAO_wind_fairlead = inputs['Re_RAO_wind_fairlead'] + 1j * inputs['Im_RAO_wind_fairlead']
		
		partials['Re_RAO_wind_moor_ten', 'Re_RAO_wind_fairlead'] = np.ones(N_omega) * K_moor
		partials['Re_RAO_wind_moor_ten', 'K_moor'] = np.real(RAO_wind_fairlead)

		partials['Im_RAO_wind_moor_ten', 'Im_RAO_wind_fairlead'] = np.ones(N_omega) * K_moor
		partials['Im_RAO_wind_moor_ten', 'K_moor'] = np.imag(RAO_wind_fairlead)