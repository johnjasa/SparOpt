import numpy as np

from openmdao.api import ExplicitComponent

class Constr50TowerStress(ExplicitComponent):

	def setup(self):
		self.add_input('maxval_tower_stress', val=np.zeros(10), units='MPa')
		self.add_input('prob_max_tower_stress', val=np.zeros(10), units='MPa')

		self.add_output('constr_50_tower_stress', val=np.zeros(10))

		self.declare_partials('constr_50_tower_stress', 'maxval_tower_stress', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('constr_50_tower_stress', 'prob_max_tower_stress', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		maxval_tower_stress = inputs['maxval_tower_stress']
		prob_max_tower_stress = inputs['prob_max_tower_stress']

		outputs['constr_50_tower_stress'] = maxval_tower_stress / prob_max_tower_stress - 1.
	
	def compute_partials(self, inputs, partials):
		maxval_tower_stress = inputs['maxval_tower_stress']
		prob_max_tower_stress = inputs['prob_max_tower_stress']

		partials['constr_50_tower_stress', 'maxval_tower_stress'] = 1. / prob_max_tower_stress
		partials['constr_50_tower_stress', 'prob_max_tower_stress'] = -maxval_tower_stress / prob_max_tower_stress**2.