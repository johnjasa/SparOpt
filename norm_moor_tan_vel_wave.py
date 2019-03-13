import numpy as np

from openmdao.api import ExplicitComponent

class NormMoorTanVelWave(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_input('phi_upper_end', val=0., units='rad')
		self.add_input('k_e_moor', val=0., units='N/m')
		self.add_input('k_g_moor', val=0., units='N/m')
		self.add_input('gen_m_moor', val=0., units='kg')
		self.add_input('gen_c_moor', val=0., units='N*s/m')

		self.add_output('Re_RAO_wave_moor_tan_vel', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_output('Im_RAO_wave_moor_tan_vel', val=np.zeros(N_omega), units='(m/s)/m')

		self.declare_partials('Re_RAO_wave_moor_tan_vel', 'Re_RAO_wave_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_moor_tan_vel', 'phi_upper_end')
		self.declare_partials('Im_RAO_wave_moor_tan_vel', 'Im_RAO_wave_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_moor_tan_vel', 'phi_upper_end')

	def compute(self, inputs, outputs):
		omega = self.omega
		
		RAO_wave_fairlead = inputs['Re_RAO_wave_fairlead'] + 1j * inputs['Im_RAO_wave_fairlead']
		phi = inputs['phi_upper_end']
		k_e = inputs['k_e_moor']
		k_g = inputs['k_g_moor']
		m = inputs['gen_m_moor']
		c = inputs['gen_c_moor']

		RAO_wave_moor_tan_vel = np.cos(phi) * 1j * omega * k_e / (-omega**2. * m + 1j * omega * c + k_g + k_e) * RAO_wave_fairlead

		outputs['Re_RAO_wave_moor_tan_vel'] = np.real(RAO_wave_moor_tan_vel)

		outputs['Im_RAO_wave_moor_tan_vel'] = np.imag(RAO_wave_moor_tan_vel)

	def compute_partials(self, inputs, partials): #TODO
		omega = self.omega
		N_omega = len(omega)

		K_moor = inputs['K_moor']

		RAO_wave_fairlead = inputs['Re_RAO_wave_fairlead'] + 1j * inputs['Im_RAO_wave_fairlead']
		
		partials['Re_RAO_wave_tan_vel', 'Re_RAO_wave_fairlead'] = np.ones(N_omega) * K_moor
		partials['Re_RAO_wave_tan_vel', 'K_moor'] = np.real(RAO_wave_fairlead)

		partials['Im_RAO_wave_tan_vel', 'Im_RAO_wave_fairlead'] = np.ones(N_omega) * K_moor
		partials['Im_RAO_wave_tan_vel', 'K_moor'] = np.imag(RAO_wave_fairlead)