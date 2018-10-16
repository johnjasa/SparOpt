import numpy as np

from openmdao.api import ExplicitComponent

class Cstruct(ExplicitComponent):

	def setup(self):
		self.add_output('C_struct', val=np.zeros((1,7)))

	def compute(self, inputs, outputs):
		outputs['C_struct'] = np.array([[0., 0., 0., 0., 0., 0., 1.]])