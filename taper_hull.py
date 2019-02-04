import numpy as np

from openmdao.api import ExplicitComponent

class TaperHull(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')

		self.add_output('taper_angle_hull', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar_p = inputs['D_spar_p']
		L_spar = inputs['L_spar']

		outputs['taper_angle_hull'] = np.arctan((D_spar_p[:-1] / 2. - D_spar_p[1:] / 2.) / L_spar)
	
	def compute_partials(self, inputs, partials):
		D_spar_p = inputs['D_spar_p']
		L_spar = inputs['L_spar']

		partials['taper_angle_hull', 'D_spar_p'] = np.zeros((10,11))
		partials['taper_angle_hull', 'L_spar'] = np.zeros((10,10))

		for i in xrange(10):
			partials['taper_angle_hull', 'D_spar_p'][i,i] += 1. / (1. + ((D_spar_p[i] / 2. - D_spar_p[i+1] / 2.) / L_spar[i])**2.) * (0.5 / L_spar[i])
			partials['taper_angle_hull', 'D_spar_p'][i,i+1] += 1. / (1. + ((D_spar_p[i] / 2. - D_spar_p[i+1] / 2.) / L_spar[i])**2.) * (-0.5 / L_spar[i])
			partials['taper_angle_hull', 'L_spar'][i,i] += 1. / (1. + ((D_spar_p[i] / 2. - D_spar_p[i+1] / 2.) / L_spar[i])**2.) * (-(D_spar_p[i] / 2. - D_spar_p[i+1] / 2.) / L_spar[i]**2.)