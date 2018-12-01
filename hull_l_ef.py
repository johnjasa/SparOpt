import numpy as np

from openmdao.api import ExplicitComponent

class HullLEf(ExplicitComponent):

	def setup(self):
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')

		self.add_output('l_ef', val=np.zeros(10), units='m')

		self.declare_partials('l_ef', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('l_ef', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('l_ef', 'l_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):

		for i in xrange(len(inputs['r_hull'])):
			l_ef = 1.56 * np.sqrt(inputs['r_hull'][i] * inputs['wt_spar_p'][i]) / (1. + 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i])

			if l_ef > inputs['l_stiff'][i]:
				outputs['l_ef'][i] = inputs['l_stiff'][i]
			else:
				outputs['l_ef'][i] = l_ef

	def compute_partials(self, inputs, partials):

		for i in xrange(len(inputs['r_hull'])):
			l_ef = 1.56 * np.sqrt(inputs['r_hull'][i] * inputs['wt_spar_p'][i]) / (1. + 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i])
			
			if l_ef > inputs['l_stiff'][i]:
				partials['l_ef', 'r_hull'][i] = 0.
				partials['l_ef', 'wt_spar_p'][i] = 0.
				partials['l_ef', 'l_stiff'][i] = 1.
			else:
				partials['l_ef', 'r_hull'][i] = 1.56 * 0.5 / np.sqrt(inputs['r_hull'][i] * inputs['wt_spar_p'][i]) / (1. + 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i]) * inputs['wt_spar_p'][i] + 1.56 * np.sqrt(inputs['r_hull'][i] * inputs['wt_spar_p'][i]) / (1. + 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i])**2. * 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i]**2.
				partials['l_ef', 'wt_spar_p'][i] = 1.56 * 0.5 / np.sqrt(inputs['r_hull'][i] * inputs['wt_spar_p'][i]) / (1. + 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i]) * inputs['r_hull'][i] - 1.56 * np.sqrt(inputs['r_hull'][i] * inputs['wt_spar_p'][i]) / (1. + 12. * inputs['wt_spar_p'][i] / inputs['r_hull'][i])**2. * 12. / inputs['r_hull'][i]
				partials['l_ef', 'l_stiff'][i] = 0.