import numpy as np

from openmdao.api import ExplicitComponent

class Constr50Fairlead(ExplicitComponent):

	def setup(self):
		self.add_input('maxval_fairlead', val=0., units='m')
		self.add_input('prob_max_fairlead', val=0., units='m')

		self.add_output('constr_50_fairlead', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		maxval_fairlead = inputs['maxval_fairlead']
		prob_max_fairlead = inputs['prob_max_fairlead']

		outputs['constr_50_fairlead'] = maxval_fairlead / prob_max_fairlead - 1.
	
	def compute_partials(self, inputs, partials):
		maxval_fairlead = inputs['maxval_fairlead']
		prob_max_fairlead = inputs['prob_max_fairlead']

		partials['constr_50_fairlead', 'maxval_fairlead'] = 1. / prob_max_fairlead
		partials['constr_50_fairlead', 'prob_max_fairlead'] = -maxval_fairlead / prob_max_fairlead**2.