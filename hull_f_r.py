import numpy as np

from openmdao.api import ExplicitComponent

class HullFR(ExplicitComponent):

	def setup(self):
		self.add_input('f_y', val=0., units='MPa')

		self.add_output('f_r', val=0., units='MPa')

		self.declare_partials('f_r', 'f_y', val=np.array([1.]))

	def compute(self, inputs, outputs):
		outputs['f_r'] = inputs['f_y'] #fabricated ring frames