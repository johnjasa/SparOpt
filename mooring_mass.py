import numpy as np

from openmdao.api import ExplicitComponent

class MooringMass(ExplicitComponent):

	def setup(self):
		self.add_input('eff_length_offset_ww', val=0., units='m')
		self.add_input('eff_length_offset_lw', val=0., units='m')
		self.add_input('mass_dens_moor', val=0., units='kg/m')

		self.add_output('M_moor', val=0., units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['M_moor'] = (inputs['eff_length_offset_ww'] + 2. * inputs['eff_length_offset_lw']) * inputs['mass_dens_moor']

	def compute_partials(self, inputs, partials):
		partials['M_moor', 'eff_length_offset_ww'] = inputs['mass_dens_moor']
		partials['M_moor', 'eff_length_offset_lw'] = 2. * inputs['mass_dens_moor']
		partials['M_moor', 'mass_dens_moor'] = (inputs['eff_length_offset_ww'] + 2. * inputs['eff_length_offset_lw'])