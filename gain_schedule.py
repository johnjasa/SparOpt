import numpy as np

from openmdao.api import ExplicitComponent

class GainSchedule(ExplicitComponent):

	def setup(self):
		self.add_input('bldpitch_0', val=0., units='rad')

		self.add_output('gain_corr_factor', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		bldpitch_0 = inputs['bldpitch_0']

		K1 = 2.0597 #found numerically by fitting quadratic curve to torque sensitivities at different above-rated wind speeds
		K2 = 0.0592

		outputs['gain_corr_factor'] = 1. / (1. + bldpitch_0 / K1 + bldpitch_0**2. / K2)

	def compute_partials(self, inputs, partials):
		bldpitch_0 = inputs['bldpitch_0']

		K1 = 2.0597
		K2 = 0.0592

		partials['gain_corr_factor', 'bldpitch_0'] = -1. / (1. + bldpitch_0 / K1 + bldpitch_0**2. / K2)**2. * (1. / K1 + 2. * bldpitch_0 / K2)