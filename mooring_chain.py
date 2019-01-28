import numpy as np

from openmdao.api import ExplicitComponent

class MooringChain(ExplicitComponent):

	def setup(self):
		self.add_input('D_moor', val=0., units='m')
		self.add_input('gamma_F_moor', val=0.)

		self.add_output('mass_dens_moor', val=0., units='kg/m')
		self.add_output('EA_moor', val=0., units='N')
		self.add_output('maxval_moor_ten', val=0., units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_moor = inputs['D_moor']
		gamma_F_moor = inputs['gamma_F_moor']

		#Min breaking strength taken from DNVGL-OS-E302
		#Mass density and axial stiffness taken from N. Barltrop "Floating Structures: A Guide for Design and Analysis"

		c = 22.3 #grade 3
		#c = 27.4 #grade 4
		#c = 30.4 #grade 4s
		#c = 32.0 #grade 5
	
		outputs['mass_dens_moor'] = 0.1875 * (1000. * D_moor)**2. / 9.80665
		outputs['EA_moor'] = 90000. * (1000. * D_moor)**2.
		outputs['maxval_moor_ten'] = 0.95 * c * (1000. * D_moor)**2. * (44. - 0.08 * (1000. * D_moor)) / gamma_F_moor #0.95 taken from DNV-OS-J103

	def compute_partials(self, inputs, partials):
		D_moor = inputs['D_moor']
		gamma_F_moor = inputs['gamma_F_moor']

		c = 22.3 #grade 3
		#c = 27.4 #grade 4
		#c = 30.4 #grade 4s
		#c = 32.0 #grade 5
	
		partials['mass_dens_moor', 'D_moor'] = 0.1875 * 2. * 1000.**2. * D_moor / 9.80665
		partials['EA_moor', 'D_moor'] = 90000. * 2. * 1000.**2. * D_moor
		partials['maxval_moor_ten', 'D_moor'] = 0.95 * (c * 2. * 1000.**2. * D_moor * (44. - 0.08 * (1000. * D_moor)) - c * (1000. * D_moor)**2. * 0.08 * 1000.) / gamma_F_moor
		partials['maxval_moor_ten', 'gamma_F_moor'] = -0.95 * c * (1000. * D_moor)**2. * (44. - 0.08 * (1000. * D_moor)) / gamma_F_moor**2.