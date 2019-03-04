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

		self.declare_partials('r_moor', 'moor_offset_x', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('r_moor', 'moor_offset_z', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('r_moor', 'moor_sig_surge_offset_x', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('r_moor', 'moor_sig_surge_offset_z', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('beta_moor', 'moor_offset_x', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('beta_moor', 'moor_offset_z', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('beta_moor', 'moor_sig_surge_offset_x', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('beta_moor', 'moor_sig_surge_offset_z', rows=np.arange(100), cols=np.arange(100))

	def compute(self, inputs, outputs):
		moor_offset_x = inputs['moor_offset_x']
		moor_offset_z = inputs['moor_offset_z']
		moor_sig_surge_offset_x = inputs['moor_sig_surge_offset_x']
		moor_sig_surge_offset_z = inputs['moor_sig_surge_offset_z']

		dx = moor_sig_surge_offset_x - moor_offset_x
		dz = moor_sig_surge_offset_z - moor_offset_z

		outputs['r_moor'] = np.sqrt(dx**2. + dz**2.)

		outputs['beta_moor'] = np.arctan(dz / dx)

	def compute_partials(self, inputs, partials): #TODO check
		moor_offset_x = inputs['moor_offset_x']
		moor_offset_z = inputs['moor_offset_z']
		moor_sig_surge_offset_x = inputs['moor_sig_surge_offset_x']
		moor_sig_surge_offset_z = inputs['moor_sig_surge_offset_z']

		dx = moor_sig_surge_offset_x - moor_offset_x
		dz = moor_sig_surge_offset_z - moor_offset_z

		partials['r_moor', 'moor_offset_x'] = -0.5 / np.sqrt(dx**2. + dz**2.) * 2. * dx
		partials['r_moor', 'moor_offset_z'] = -0.5 / np.sqrt(dx**2. + dz**2.) * 2. * dz
		partials['r_moor', 'moor_sig_surge_offset_x'] = 0.5 / np.sqrt(dx**2. + dz**2.) * 2. * dx
		partials['r_moor', 'moor_sig_surge_offset_z'] = 0.5 / np.sqrt(dx**2. + dz**2.) * 2. * dz

		partials['beta_moor', 'moor_offset_x'] = 1. / (1. + (dz / dx)**2.) * dz / dx**2.
		partials['beta_moor', 'moor_offset_z'] = 1. / (1. + (dz / dx)**2.) * 1. / dx
		partials['beta_moor', 'moor_sig_surge_offset_x'] = -1. / (1. + (dz / dx)**2.) * dz / dx**2.
		partials['beta_moor', 'moor_sig_surge_offset_z'] = 1. / (1. + (dz / dx)**2.) * 1. / dx