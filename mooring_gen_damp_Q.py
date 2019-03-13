import numpy as np

from openmdao.api import ExplicitComponent

class MooringGenDampQ(ExplicitComponent):

	def setup(self):
		self.add_input('gen_c_moor', val=0., units='N*s/m')
		self.add_input('stddev_moor_tan_vel', val=0., units='m/s')

		self.add_output('gen_c_moor_Q', val=0., units='N*s**2/m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		outputs['gen_c_moor_Q'] = np.sqrt(np.pi / 8.) * inputs['gen_c_moor'] / inputs['stddev_moor_tan_vel']

	def compute_partials(self, inputs, partials): #TODO

		partials['gen_c_moor', 'norm_r_moor'] = 0.