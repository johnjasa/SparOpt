import numpy as np

from openmdao.api import ExplicitComponent

class WindSpeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('blades', types=dict)
		self.options.declare('freqs', types=dict)

	def setup(self):
		blades = self.options['blades']
		self.N_b_elem = blades['N_b_elem']
		self.windfolder = blades['windfolder']

		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('windspeed_0', val=0., units='m/s')
		self.add_input('rotspeed_0', val=0., units='rad/s')
		self.add_input('b_elem_r', val=np.zeros(self.N_b_elem), units='m')
		self.add_input('b_elem_dr', val=0., units='m')
		self.add_input('dFn_dv', val=np.zeros(self.N_b_elem), units='N*s/m**2')
		self.add_input('dFt_dv', val=np.zeros(self.N_b_elem), units='N*s/m**2')

		self.add_output('thrust_wind', val=np.zeros(N_omega), units='m/s')
		self.add_output('moment_wind', val=np.zeros(N_omega), units='m/s')
		self.add_output('torque_wind', val=np.zeros(N_omega), units='m/s')

	def compute(self,inputs,outputs):
		omega = self.omega
		Vhub = inputs['windspeed_0']

		omega_ws, thrust_wind, moment_wind, torque_wind = np.loadtxt(self.windfolder + 'eq_wind_%d.dat' % Vhub, unpack=True)
		outputs['thrust_wind'] = np.interp(omega,omega_ws,thrust_wind)
		outputs['moment_wind'] = np.interp(omega,omega_ws,moment_wind)
		outputs['torque_wind'] = np.interp(omega,omega_ws,torque_wind)