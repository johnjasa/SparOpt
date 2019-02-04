import numpy as np

from openmdao.api import ExplicitComponent

class TaperTower(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')

		self.add_output('taper_angle_tower', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower_p = inputs['D_tower_p']
		L_tower = inputs['L_tower']

		outputs['taper_angle_tower'] = np.arctan((D_tower_p[:-1] / 2. - D_tower_p[1:] / 2.) / L_tower)
	
	def compute_partials(self, inputs, partials):
		D_tower_p = inputs['D_tower_p']
		L_tower = inputs['L_tower']

		partials['taper_angle_tower', 'D_tower_p'] = np.zeros((10,11))
		partials['taper_angle_tower', 'L_tower'] = np.zeros((10,10))

		for i in xrange(10):
			partials['taper_angle_tower', 'D_tower_p'][i,i] += 1. / (1. + ((D_tower_p[i] / 2. - D_tower_p[i+1] / 2.) / L_tower[i])**2.) * (0.5 / L_tower[i])
			partials['taper_angle_tower', 'D_tower_p'][i,i+1] += 1. / (1. + ((D_tower_p[i] / 2. - D_tower_p[i+1] / 2.) / L_tower[i])**2.) * (-0.5 / L_tower[i])
			partials['taper_angle_tower', 'L_tower'][i,i] += 1. / (1. + ((D_tower_p[i] / 2. - D_tower_p[i+1] / 2.) / L_tower[i])**2.) * (-(D_tower_p[i] / 2. - D_tower_p[i+1] / 2.) / L_tower[i]**2.)