import numpy as np

from openmdao.api import ExplicitComponent

class MooringSurgeDamp(ExplicitComponent):

	def setup(self):
		self.add_input('k_e_moor', val=0., units='N/m')
		self.add_input('k_g_moor', val=0., units='N/m')
		self.add_input('gen_c_moor', val=0., units='N*s/m')
		self.add_input('stddev_surge_vel_WF', val=0., units='m/s')
		self.add_input('phi_upper_end', val=0., units='rad')

		self.add_output('moor_surge_damp', val=0., units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		k_e = inputs['k_e_moor']
		k_g = inputs['k_g_moor']
		c = inputs['gen_c_moor']
		stddev_surge_vel_WF = inputs['stddev_surge_vel_WF']
		phi_upper_end = inputs['phi_upper_end']

		outputs['moor_surge_damp'] = c * np.sqrt(8. / np.pi) * stddev_surge_vel_WF * 1. / (1. + k_g / k_e) * np.cos(phi_upper_end)**2. * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)

	def compute_partials(self, inputs, partials): #TODO check
		k_e = inputs['k_e_moor']
		k_g = inputs['k_g_moor']
		c = inputs['gen_c_moor']
		stddev_surge_vel_WF = inputs['stddev_surge_vel_WF']
		phi_upper_end = inputs['phi_upper_end']

		partials['moor_surge_damp', 'k_e_moor'] = c * np.sqrt(8. / np.pi) * stddev_surge_vel_WF * 1. / (1. + k_g / k_e)**2. * k_g / k_e**2. * np.cos(phi_upper_end)**2. * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)
		partials['moor_surge_damp', 'k_g_moor'] = -c * np.sqrt(8. / np.pi) * stddev_surge_vel_WF * 1. / (1. + k_g / k_e)**2. * 1. / k_e * np.cos(phi_upper_end)**2. * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)
		partials['moor_surge_damp', 'gen_c_moor'] = np.sqrt(8. / np.pi) * stddev_surge_vel_WF * 1. / (1. + k_g / k_e) * np.cos(phi_upper_end)**2. * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)
		partials['moor_surge_damp', 'stddev_surge_vel_WF'] = c * np.sqrt(8. / np.pi) * 1. / (1. + k_g / k_e) * np.cos(phi_upper_end)**2. * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)
		partials['moor_surge_damp', 'phi_upper_end'] = -c * np.sqrt(8. / np.pi) * stddev_surge_vel_WF * 1. / (1. + k_g / k_e) * 2. * np.cos(phi_upper_end) * np.sin(phi_upper_end) * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)