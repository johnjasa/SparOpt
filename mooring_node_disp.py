import numpy as np

from openmdao.api import ExplicitComponent

class MooringNodeDisp(ExplicitComponent):

	def setup(self):
		self.add_input('moor_offset_x', val=np.zeros(100), units='m')
		self.add_input('moor_offset_z', val=np.zeros(100), units='m')
		self.add_input('moor_sig_surge_offset_x', val=np.zeros(100), units='m')
		self.add_input('moor_sig_surge_offset_z', val=np.zeros(100), units='m')

		self.add_output('r_moor', val=np.zeros(100), units='m')
		self.add_output('beta_moor', val=np.zeros(100), units='rad')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		moor_offset_x = inputs['moor_offset_x']
		moor_offset_z = inputs['moor_offset_z']
		moor_sig_surge_offset_x = inputs['moor_sig_surge_offset_x']
		moor_sig_surge_offset_z = inputs['moor_sig_surge_offset_z']

		dx = moor_sig_surge_offset_x - moor_offset_x
		dz = moor_sig_surge_offset_z - moor_offset_z

		outputs['r_moor'] = np.sqrt(dx**2. + dz**2.)

		outputs['beta_moor'] = np.arctan(dz / dx)

	def compute_partials(self, inputs, partials): #TODO
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']