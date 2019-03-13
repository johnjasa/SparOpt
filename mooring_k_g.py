import numpy as np

from openmdao.api import ExplicitComponent

class MooringKg(ExplicitComponent):

	def setup(self):
		#self.add_input('mean_moor_ten', val=0., units='N')
		#self.add_input('moor_ten_sig_surge_tot', val=0., units='N')
		self.add_input('sig_tan_motion', val=0., units='m')
		self.add_input('mass_dens_moor', val=0., units='kg/m')
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('eff_length_offset_ww', val=0., units='m')
		self.add_input('moor_tension_sig_surge_offset', val=0., units='N')
		self.add_input('eff_length_sig_surge_offset', val=0., units='m')

		self.add_output('k_g_moor', val=0., units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		w = inputs['mass_dens_moor'] * 9.80665
		T0 = inputs['moor_tension_offset_ww'] + w * inputs['eff_length_offset_ww']
		T1 = inputs['moor_tension_sig_surge_offset'] + w * inputs['eff_length_sig_surge_offset']
		dx = inputs['sig_tan_motion']

		outputs['k_g_moor'] = (T1 - T0) / dx

	def compute_partials(self, inputs, partials): #TODO check
		T0 = inputs['mean_moor_ten']
		T1 = inputs['moor_ten_sig_surge_tot']
		dx = inputs['sig_tan_motion']

		partials['k_g_moor', 'mean_moor_ten'] = -1. / dx
		partials['k_g_moor', 'moor_ten_sig_surge_tot'] = 1. / dx
		partials['k_g_moor', 'sig_tan_motion'] = -(T1 - T0) / dx**2.