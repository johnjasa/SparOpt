import numpy as np

from openmdao.api import ExplicitComponent

class LowerBoundZMoor(ExplicitComponent):

	def setup(self):
		self.add_input('z_moor', val=0., units='m')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('lower_bound_z_moor', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['lower_bound_z_moor'] = -inputs['spar_draft'] / inputs['z_moor'] - 1.

	def compute_partials(self, inputs, partials):
		partials['lower_bound_z_moor', 'spar_draft'] = -1. / inputs['z_moor']
		partials['lower_bound_z_moor', 'z_moor'] = inputs['spar_draft'] / inputs['z_moor']**2.