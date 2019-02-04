import numpy as np

from openmdao.api import ExplicitComponent

class CoBCoG(ExplicitComponent):

	def setup(self):
		self.add_input('CoB', val=0., units='m')
		self.add_input('CoG_total', val=0., units='m')

		self.add_output('CoB_CoG', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['CoB_CoG'] = inputs['CoG_total'] / inputs['CoB'] - 1.

	def compute_partials(self, inputs, partials):
		partials['CoB_CoG', 'CoG_total'] = 1. / inputs['CoB']
		partials['CoB_CoG', 'CoB'] = -inputs['CoG_total'] / inputs['CoB']**2.