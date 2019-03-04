import numpy as np

from openmdao.api import ExplicitComponent

class MooringOffsetCoords(ExplicitComponent):

	def setup(self):
		self.add_input('mass_dens_moor', val=0., units='kg/m')
		self.add_input('moor_tension_offset_ww', val=0., units='N')
		self.add_input('eff_length_offset_ww', val=0., units='m')
		self.add_input('EA_moor', val=0., units='N')

		self.add_output('moor_offset_x', val=np.zeros(100), units='m')
		self.add_output('moor_offset_z', val=np.zeros(100), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']
		EA = inputs['EA_moor']

		w = mu * 9.80665

		s = np.linspace(0.,L,100)

		x = H / w * np.arcsinh(w * s / H) + H * s / EA
		x = x - x[-1]
		
		z = w * s**2. / (2 * EA) + H / w * (np.sqrt(1. + (w * s / H)**2.) - 1.)

		outputs['moor_offset_x'] = x
		outputs['moor_offset_z'] = z

	def compute_partials(self, inputs, partials): #TODO check
		mu = inputs['mass_dens_moor']
		H = inputs['moor_tension_offset_ww']
		L = inputs['eff_length_offset_ww']
		EA = inputs['EA_moor']

		w = mu * 9.80665

		s = np.linspace(0.,L,100)

		dL = s[1] - s[0]

		x = H / w * np.arcsinh(w * s / H) + H * s / EA
		x = x - x[-1]
		
		z = w * s**2. / (2. * EA) + H / w * (np.sqrt(1. + (w * s / H)**2.) - 1.)

		partials['moor_offset_z', 'mass_dens_moor'] = H / 9.80665 * np.arcsinh(w * s / H) + H / w * 1. / np.sqrt((w * s / H)**2. + 1.) * 9.80665 * s / H - H / 9.80665 * np.arcsinh(w * L / H) - H / w * 1. / np.sqrt((w * L / H)**2. + 1.) * 9.80665 * L / H
		partials['moor_offset_z', 'moor_tension_offset_ww'] = 1. / w * np.arcsinh(w * s / H) + H / w * 1. / np.sqrt((w * s / H)**2. + 1) * 2. * (w * s / H) * w / H + s / EA - 1. / w * np.arcsinh(w * L / H) - H / w * 1. / np.sqrt((w * L / H)**2. + 1) * 2. * (w * L / H) * w / H - L / EA
		partials['moor_offset_z', 'eff_length_offset_ww'] = H / w * 1. / np.sqrt((w * s / H)**2. + 1) * w / H * dL + H * dL / EA - H / w * 1. / np.sqrt((w * L / H)**2. + 1) * w / H - H / EA
		partials['moor_offset_z', 'EA_moor'] = -H * s / EA**2. + H * L / EA**2.

		partials['moor_offset_z', 'mass_dens_moor'] = 9.80665 * s**2. / (2 * EA) - H / w**2. * 9.80665 * (np.sqrt(1. + (w * s / H)**2.) - 1.) + H / w * 0.5 / np.sqrt(1. + (w * s / H)**2.) * 2. * (w * s / H) * 9.80665 * s / H
		partials['moor_offset_z', 'moor_tension_offset_ww'] = 1. / w * (np.sqrt(1. + (w * s / H)**2.) - 1.) - H / w * 0.5 / np.sqrt(1. + (w * s / H)**2.) * 2. * (w * s / H) * w * s / H**2.
		partials['moor_offset_z', 'eff_length_offset_ww'] = 2. * w * s * dL / (2. * EA) + H / w * 0.5 / np.sqrt(1. + (w * s / H)**2.) * 2. * (w * s / H) * w / H * dL
		partials['moor_offset_z', 'EA_moor'] = -2. * w * s**2. / (2. * EA)**2.