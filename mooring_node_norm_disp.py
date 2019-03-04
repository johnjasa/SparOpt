import numpy as np

from openmdao.api import ExplicitComponent

class MooringNodeNormDisp(ExplicitComponent):

	def setup(self):
		self.add_input('sig_tan_motion', val=0., units='m')
		self.add_input('phi_moor', val=np.zeros(100), units='rad')
		self.add_input('r_moor', val=np.zeros(100), units='m')
		self.add_input('beta_moor', val=np.zeros(100), units='rad')

		self.add_output('norm_r_moor', val=np.zeros(100), units='m')

		self.declare_partials('norm_r_moor', 'sig_tan_motion')
		self.declare_partials('norm_r_moor', 'phi_moor', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('norm_r_moor', 'r_moor', rows=np.arange(100), cols=np.arange(100))
		self.declare_partials('norm_r_moor', 'beta_moor', rows=np.arange(100), cols=np.arange(100))

	def compute(self, inputs, outputs):
		sig_tan_motion = inputs['sig_tan_motion']
		phi = inputs['phi_moor']
		r = inputs['r_moor']
		beta = inputs['beta_moor']

		theta = phi - beta

		norm_r = r * np.sin(theta)
		norm_r = norm_r / sig_tan_motion

		outputs['norm_r_moor'] = norm_r

	def compute_partials(self, inputs, partials): #TODO check
		sig_tan_motion = inputs['sig_tan_motion']
		phi = inputs['phi_moor']
		r = inputs['r_moor']
		beta = inputs['beta_moor']

		theta = phi - beta

		norm_r = r * np.sin(theta) / sig_tan_motion

		partials['norm_r_moor', 'sig_tan_motion'] = -r * np.sin(theta) / sig_tan_motion**2.
		partials['norm_r_moor', 'phi_moor'] = r * np.cos(phi - beta) / sig_tan_motion
		partials['norm_r_moor', 'r_moor'] = np.sin(theta) / sig_tan_motion
		partials['norm_r_moor', 'beta_moor'] = -r * np.cos(phi - beta) / sig_tan_motion