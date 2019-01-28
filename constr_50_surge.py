import numpy as np

from openmdao.api import ExplicitComponent

class Constr50Surge(ExplicitComponent):

	def setup(self):
		self.add_input('maxval_surge', val=0., units='m')
		self.add_input('prob_max_surge', val=0., units='m')

		self.add_output('constr_50_surge', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		maxval_surge = inputs['maxval_surge']
		prob_max_surge = inputs['prob_max_surge']

		outputs['constr_50_surge'] = maxval_surge / prob_max_surge - 1.
	
	def compute_partials(self, inputs, partials):
		maxval_surge = inputs['maxval_surge']
		prob_max_surge = inputs['prob_max_surge']

		partials['constr_50_surge', 'maxval_surge'] = 1. / prob_max_surge
		partials['constr_50_surge', 'prob_max_surge'] = -maxval_surge / prob_max_surge**2.