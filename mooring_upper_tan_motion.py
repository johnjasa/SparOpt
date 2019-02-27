import numpy as np

from openmdao.api import ExplicitComponent

class MooringUpperTanMotion(ExplicitComponent):

	def setup(self):
		self.add_input('stddev_surge_WF', val=0., units='m')
		self.add_input('phi_upper_end', val=0., units='rad')

		self.add_output('sig_tan_motion', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		stddev_surge_WF = inputs['stddev_surge_WF']
		phi = inputs['phi_upper_end']
	
		outputs['sig_tan_motion'] = 2. * stddev_surge_WF * np.cos(phi)

	def compute_partials(self, inputs, partials): #TODO check
		stddev_surge_WF = inputs['stddev_surge_WF']
		phi = inputs['phi_upper_end']
	
		partials['sig_tan_motion', 'stddev_surge_WF'] = 2. * np.cos(phi)
		partials['sig_tan_motion', 'phi_upper_end'] = -2. * stddev_surge_WF * np.sin(phi)