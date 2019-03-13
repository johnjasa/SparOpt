import numpy as np

from openmdao.api import ExplicitComponent

class MooringCost(ExplicitComponent):

	def setup(self):
		self.add_input('len_tot_moor', val=0., units='m')
		self.add_input('mass_dens_moor', val=0., units='kg/m')

		self.add_output('mooring_cost', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		k_m = 3.5 * 1.15 #3-4.5 euro per kg (ref. email Kjell Larsen) times dollar per euro

		outputs['mooring_cost'] = 3. * k_m * inputs['len_tot_moor'] * inputs['mass_dens_moor'] #total for three mooring lines


	def compute_partials(self, inputs, partials):

		k_m = 3.5 * 1.15 #3-4.5 euro per kg (ref. email Kjell Larsen) times dollar per euro

		partials['mooring_cost', 'len_tot_moor'] = 3. * k_m * inputs['mass_dens_moor']
		partials['mooring_cost', 'mass_dens_moor'] = 3. * k_m * inputs['len_tot_moor']