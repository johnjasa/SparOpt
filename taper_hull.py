import numpy as np

from openmdao.api import ExplicitComponent

class TaperHull(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')

		self.add_output('taper_hull', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar_p = inputs['D_spar_p']

		outputs['taper_hull'] = D_spar_p[1:] / D_spar_p[:-1]
	
	def compute_partials(self, inputs, partials):
		D_spar_p = inputs['D_spar_p']

		for i in xrange(10):
			partials['taper_hull', 'D_spar_p'][i,i] = -D_spar_p[i+1] / D_spar_p[i]**2.
			partials['taper_hull', 'D_spar_p'][i,i+1] = 1. / D_spar_p[i]