import numpy as np

from openmdao.api import ExplicitComponent

class MooringSigSurgeOffsetCoords(ExplicitComponent):

	def setup(self):
		self.add_input('mass_dens_moor', val=0., units='m')
		self.add_input('moor_tension_sig_surge_offset', val=0., units='N')
		self.add_input('eff_length_sig_surge_offset', val=0., units='m')
		self.add_input('EA_moor', val=0., units='N')

		self.add_output('moor_sig_surge_offset_x', val=np.zeros(100), units='m')
		self.add_output('moor_sig_surge_offset_z', val=np.zeros(100), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_sig_surge_offset']
		L = inputs['eff_length_sig_surge_offset']
		EA = inputs['EA_moor']

		w = mu * 9.80665

		s = np.linspace(0.,L,100)

		x = H / w * np.arcsinh(w * s / H) + H * s / EA
		x = x - x[-1]
		
		z = w * s**2. / (2 * EA) + H / w * (np.sqrt(1. + (w * s / H)**2.) - 1.)

		outputs['moor_sig_surge_offset_x'] = x
		outputs['moor_sig_surge_offset_z'] = z

	def compute_partials(self, inputs, partials): #TODO
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_sig_surge_offset']
		L = inputs['eff_length_sig_surge_offset']
		EA = inputs['EA_moor']