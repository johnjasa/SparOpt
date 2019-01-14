import numpy as np

from openmdao.api import ExplicitComponent

class TowerCx(ExplicitComponent):

	def setup(self):
		self.add_input('tower_omega', val=np.zeros(10))
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')

		self.add_output('C_x', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		w = inputs['tower_omega']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		r = (D_tower_p[:-1] - wt_tower_p[:-1]) / 2.

		C_xb = 6. #clamped-clamped

		for i in xrange(10):
			if (1. + 0.2 / C_xb * (1. - 2. * w[i] * wt_tower_p[i] / r[i])) > 0.6:
				C_xN = 1. + 0.2 / C_xb * (1. - 2. * w[i] * wt_tower_p[i] / r[i])
			else:
				C_xN = 0.6

			if w[i] <= 1.7:
				C_x = 1.36 - 1.83 / w[i] + 2.07 / w[i]**2.
			elif w[i] > 0.5 * r[i] / wt_tower_p[i]:
				C_x = C_xN
			else:
				C_x = 1.

			outputs['C_x'][i] = C_x

	def compute_partials(self, inputs, partials):
		w = inputs['tower_omega']
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		r = (D_tower_p[:-1] - wt_tower_p[:-1]) / 2.

		C_xb = 6. #clamped-clamped

		partials['C_x', 'tower_omega'] = np.zeros((10,10))
		partials['C_x', 'D_tower_p'] = np.zeros((10,11))
		partials['C_x', 'wt_tower_p'] = np.zeros((10,11))

		for i in xrange(10):
			if w[i] <= 1.7:
				partials['C_x', 'tower_omega'][i,i] += 1.83 / w[i]**2. - 2. * 2.07 / w[i]**3.
			elif w[i] > 0.5 * r[i] / wt_tower_p[i]:
				if (1. + 0.2 / C_xb * (1. - 2. * w[i] * wt_tower_p[i] / r[i])) > 0.6:
					partials['C_x', 'tower_omega'][i,i] += -0.2 / C_xb * 2. * wt_tower_p[i] / r[i]
					partials['C_x', 'D_tower_p'][i,i] += 0.2 / C_xb * 2. * w[i] * wt_tower_p[i] / r[i]**2. * 0.5
					partials['C_x', 'wt_tower_p'][i,i] += -0.2 / C_xb * 2. * w[i] * (1. / r[i] + wt_tower_p[i] / r[i]**2. * 0.5)
				else:
					continue
			else:
				continue

			