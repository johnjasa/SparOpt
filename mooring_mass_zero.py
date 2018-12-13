import numpy as np

from openmdao.api import ExplicitComponent

class MooringMassZero(ExplicitComponent):

	def setup(self):
		self.add_input('eff_length_zero', val=0., units='m')
		self.add_input('mass_dens_moor', val=0., units='kg/m')

		self.add_output('M_moor_zero', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['M_moor_zero'] = 3. * inputs['eff_length_zero'] * inputs['mass_dens_moor']

	def compute_partials(self, inputs, partials):
		partials['M_moor_zero', 'eff_length_zero'] = 3. * inputs['mass_dens_moor']
		partials['M_moor_zero', 'mass_dens_moor'] = 3. * inputs['eff_length_zero']