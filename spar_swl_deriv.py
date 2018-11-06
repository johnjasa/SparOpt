import numpy as np

from openmdao.api import ExplicitComponent

class SparSWLDeriv(ExplicitComponent):

	def setup(self):
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_d_sparnode', val=np.zeros(14), units='m/m')

		self.add_output('x_d_swl', val=0., units='m/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z_SWL = 0.
		SWLidx = np.concatenate(np.where(inputs['z_sparnode']==z_SWL))

		outputs['x_d_swl'] = inputs['x_d_sparnode'][SWLidx]

	def compute_partials(self, inputs, partials):
		z_SWL = 0.
		SWLidx = np.concatenate(np.where(inputs['z_sparnode']==z_SWL))

		partials['x_d_swl', 'x_d_sparnode'] = np.zeros((1,14))
		partials['x_d_swl', 'x_d_sparnode'][0,SWLidx] = 1.