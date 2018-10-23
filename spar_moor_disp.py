import numpy as np

from openmdao.api import ExplicitComponent

class SparMoorDisp(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_moor', val=0., units='m')

		self.add_output('x_moor', val=0., units='m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		mooridx = np.concatenate(np.where(inputs['z_sparnode']==inputs['z_moor'][0]))

		outputs['x_moor'] = inputs['x_sparnode'][mooridx]