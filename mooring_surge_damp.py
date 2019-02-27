import numpy as np

from openmdao.api import ExplicitComponent

class MooringSurgeDamp(ExplicitComponent):

	def setup(self):
		self.add_input('k_e_moor', val=0., units='N/m')
		self.add_input('k_g_moor', val=0., units='N/m')
		self.add_input('gen_c_moor', val=0., units='N*s/m')
		self.add_input('stddev_surge_LF', val=0., units='m')
		self.add_input('phi_upper_end', val=0., units='rad')

		self.add_output('moor_surge_damp', val=0., units='N*s/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		k_e = inputs['k_e_moor']
		k_g = inputs['k_g_moor']
		c = inputs['gen_c_moor']
		stddev_surge_LF = inputs['stddev_surge_LF']
		phi_upper_end = inputs['phi_upper_end']

		outputs['moor_surge_damp'] = c * np.sqrt(8. / np.pi) * stddev_surge_LF * 1. / (1. + k_g / k_e) * np.cos(phi_upper_end)**2. * (1. + 2. * np.cos(60. * np.pi / 180.)**2.)

	def compute_partials(self, inputs, partials): #TODO
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']