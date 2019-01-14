import numpy as np

from openmdao.api import ExplicitComponent

class TowerOmega(ExplicitComponent):

	def setup(self):
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')

		self.add_output('tower_omega', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		l = inputs['L_tower']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		r = (D_tower_p[:-1] - wt_tower_p[:-1]) / 2.

		outputs['tower_omega'] = l / np.sqrt(r * wt_tower_p[:-1])

	def compute_partials(self, inputs, partials):
		l = inputs['L_tower']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		r = (D_tower_p[:-1] - wt_tower_p[:-1]) / 2.

		partials['tower_omega', 'L_tower'] = np.zeros((10,10))
		partials['tower_omega', 'D_tower_p'] = np.zeros((10,11))
		partials['tower_omega', 'wt_tower_p'] = np.zeros((10,11))

		for i in xrange(10):
			partials['tower_omega', 'L_tower'][i,i] += 1. / np.sqrt(r[i] * wt_tower_p[i])
			partials['tower_omega', 'D_tower_p'][i,i] += -0.5 * l[i] / (r[i] * wt_tower_p[i])**(3. / 2.) * 0.5 * wt_tower_p[i]
			partials['tower_omega', 'wt_tower_p'][i,i] += -0.5 * l[i] / (r[i] * wt_tower_p[i])**(3. / 2.) * (r[i] - 0.5 * wt_tower_p[i])