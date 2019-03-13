import numpy as np

from openmdao.api import ExplicitComponent

class MoorKFactor(ExplicitComponent):

	def setup(self):
		self.add_input('stddev_moor_v', val=0., units='N')
		self.add_input('stddev_moor_tan_vel', val=0., units='m/s')
		self.add_input('gen_c_moor_Q', val=0., units='N*s**2/m**2')

		self.add_output('moor_k_factor', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		outputs['moor_k_factor'] = inputs['gen_c_moor_Q'] * inputs['stddev_moor_tan_vel']**2. / inputs['stddev_moor_v']

	def compute_partials(self, inputs, partials):

		partials['stddev_moor_tan_vel', 'resp_moor_tan_vel'] = 0.