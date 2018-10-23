import numpy as np

from openmdao.api import ExplicitComponent

class SparAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')

		self.add_output('A11', val=0., units='kg')
		self.add_output('A55', val=0., units='kg*m**2')
		self.add_output('A15', val=0., units='kg*m')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']

		#TODO: rewrite completely

		outputs['A11'] = 1025. * np.pi/4. * (D_spar[0]**2. * 108. + D_spar[-2]**2. * 8. + D_spar[-1]**2. * 4.)
		outputs['A15'] = 1025. * np.pi/4. * (D_spar[-1]**2. * (-0.5) * 4.**2. - 0.5 * D_spar[-2]**2. * (12.**2. - (-12.+8.)**2.) - 0.5 * D_spar[0]**2. * ((12.+108.)**2. - (-(12.+108.)+108.)**2.))
		outputs['A55'] = 1025. * np.pi/4. * (D_spar[-1]**2. * 1. / 3. * 4.**3. + D_spar[-2]**2. * 8. * (1./12. * 8.**2. + (0.5*(-2.*12.+8.))**2.) + D_spar[0]**2. * 108. * (1./12. * 108.**2. + (0.5*(-2.*(12.+108.)+108.))**2.))