import numpy as np

from openmdao.api import ExplicitComponent

class Constr50MoorTen(ExplicitComponent):

	def setup(self):
		self.add_input('maxval_moor_ten', val=0., units='N')
		self.add_input('prob_max_moor_ten', val=0., units='N')

		self.add_output('constr_50_moor_ten', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		maxval_moor_ten = inputs['maxval_moor_ten']
		prob_max_moor_ten = inputs['prob_max_moor_ten']

		outputs['constr_50_moor_ten'] = maxval_moor_ten / prob_max_moor_ten - 1.
	
	def compute_partials(self, inputs, partials):
		maxval_moor_ten = inputs['maxval_moor_ten']
		prob_max_moor_ten = inputs['prob_max_moor_ten']

		partials['constr_50_moor_ten', 'maxval_moor_ten'] = 1. / prob_max_moor_ten
		partials['constr_50_moor_ten', 'prob_max_moor_ten'] = -maxval_moor_ten / prob_max_moor_ten**2.