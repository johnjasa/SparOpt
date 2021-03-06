import numpy as np

from openmdao.api import ExplicitComponent

class Ccontrl(ExplicitComponent):

	def setup(self):
		self.add_input('windspeed_0', val=0., units='m/s')
		self.add_input('rotspeed_0', val=0., units='rad/s')
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)

		self.add_output('C_contrl', val=np.zeros((2,2)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		windspeed_0 = inputs['windspeed_0']
		rotspeed_0 = inputs['rotspeed_0']
		k_i = inputs['k_i']
		k_p = inputs['k_p']
		eta = inputs['gain_corr_factor']

		K = 251153.48 #Nm/(rad/s)**2

		if windspeed_0 < 11.4: #rated wind speed
			outputs['C_contrl'] = np.array([[0., 2. * K * rotspeed_0],[0., 0.]])
		else:
			outputs['C_contrl'] = np.array([[0., 0.],[eta * k_i, eta * k_p]])

	def compute_partials(self, inputs, partials):
		windspeed_0 = inputs['windspeed_0']
		rotspeed_0 = inputs['rotspeed_0']
		k_i = inputs['k_i']
		k_p = inputs['k_p']
		eta = inputs['gain_corr_factor']

		K = 251153.48 #Nm/(rad/s)**2

		if windspeed_0 < 11.4: #rated wind speed
			partials['C_contrl', 'windspeed_0'] = np.array([[0., 0.],[0., 0.]])
			partials['C_contrl', 'rotspeed_0'] = np.array([[0., 2. * K],[0., 0.]])
			partials['C_contrl', 'k_i'] = np.array([[0., 0.],[0., 0.]])
			partials['C_contrl', 'k_p'] = np.array([[0., 0.],[0., 0.]])
			partials['C_contrl', 'gain_corr_factor'] = np.array([[0., 0.],[0., 0.]])
		else:
			partials['C_contrl', 'windspeed_0'] = np.array([[0., 0.],[0., 0.]])
			partials['C_contrl', 'rotspeed_0'] = np.array([[0., 0.],[0., 0.]])
			partials['C_contrl', 'k_i'] = np.array([[0., 0.],[eta, 0.]])
			partials['C_contrl', 'k_p'] = np.array([[0., 0.],[0., eta]])
			partials['C_contrl', 'gain_corr_factor'] = np.array([[0., 0.],[k_i, k_p]])