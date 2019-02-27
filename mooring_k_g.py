import numpy as np

from openmdao.api import ExplicitComponent

class MooringKg(ExplicitComponent):

	def setup(self):
		self.add_input('mean_moor_ten', val=0., units='N')
		self.add_input('moor_ten_sig_surge_tot', val=0., units='N')
		self.add_input('sig_tan_motion', val=0., units='m')

		self.add_output('k_g_moor', val=0., units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		T0 = inputs['mean_moor_ten']
		T1 = inputs['moor_ten_sig_surge_tot']
		dx = inputs['sig_tan_motion']

		outputs['k_g_moor'] = (T1 - T0) / dx

	def compute_partials(self, inputs, partials): #TODO check
		T0 = inputs['mean_moor_ten']
		T1 = inputs['moor_ten_sig_surge_tot']
		dx = inputs['sig_tan_motion']

		partials['k_g_moor', 'mean_moor_ten'] = -1. / dx
		partials['k_g_moor', 'moor_ten_sig_surge_tot'] = 1. / dx
		partials['k_g_moor', 'sig_tan_motion'] = -(T1 - T0) / dx**2.