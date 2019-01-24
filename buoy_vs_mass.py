import numpy as np

from openmdao.api import ExplicitComponent

class BuoyVsMass(ExplicitComponent):

	def setup(self):
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('tot_M_spar', val=0., units='kg')
		self.add_input('M_turb', val=0., units='kg')
		self.add_input('M_moor_zero', val=0., units='kg')

		self.add_output('buoy_mass', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		buoy_spar = inputs['buoy_spar']
		tot_M_spar = inputs['tot_M_spar']
		M_turb = inputs['M_turb']
		M_moor_zero = inputs['M_moor_zero']

		outputs['buoy_mass'] = buoy_spar / 9.80665 - (tot_M_spar + M_turb + M_moor_zero)

	def compute_partials(self, inputs, partials):
		buoy_spar = inputs['buoy_spar'][0]
		tot_M_spar = inputs['tot_M_spar'][0]
		M_turb = inputs['M_turb'][0]
		M_moor_zero = inputs['M_moor_zero'][0]

		partials['buoy_mass', 'buoy_spar'] = 1. / 9.80665
		partials['buoy_mass', 'tot_M_spar'] = -1.
		partials['buoy_mass', 'M_turb'] = -1.
		partials['buoy_mass', 'M_moor_zero'] = -1.