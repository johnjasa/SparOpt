import numpy as np

from openmdao.api import ExplicitComponent

class StdDevMoorV(ExplicitComponent):

	def setup(self):
		self.add_input('stddev_moor_ten_dyn', val=0., units='N')
		self.add_input('stddev_moor_tan_vel', val=0., units='m/s')
		self.add_input('gen_c_moor_Q', val=0., units='N*s**2/m**2')

		self.add_output('stddev_moor_v', val=0., units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		outputs['stddev_moor_v'] = np.sqrt(inputs['stddev_moor_ten_dyn']**2. - 3. * (inputs['gen_c_moor_Q'] * inputs['stddev_moor_tan_vel']**2.)**2.)

	def compute_partials(self, inputs, partials):

		partials['stddev_moor_tan_vel', 'resp_moor_tan_vel'] = 0.