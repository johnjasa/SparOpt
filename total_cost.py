import numpy as np

from openmdao.api import ExplicitComponent

class TotalCost(ExplicitComponent):

	def setup(self):
		self.add_input('spar_cost', val=0.)
		self.add_input('tower_cost', val=0.)
		self.add_input('mooring_cost', val=0.)

		self.add_output('total_cost', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		outputs['total_cost'] = inputs['spar_cost'] + inputs['tower_cost'] + inputs['mooring_cost']

	def compute_partials(self, inputs, partials):

		partials['total_cost', 'spar_cost'] = 1.
		partials['total_cost', 'tower_cost'] = 1.
		partials['total_cost', 'mooring_cost'] = 1.