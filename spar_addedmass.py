import numpy as np

from openmdao.api import ExplicitComponent

class SparAddedMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('L_secs', val=np.zeros(3), units='m')

		self.add_output('A11', val=0.)
		self.add_output('A55', val=0.)
		self.add_output('A15', val=0.)

	def compute(self, inputs, outputs):
		D_secs = inputs['D_secs']
		L_secs = inputs['L_secs']

		outputs['A11'] = 1025. * np.pi/4. * (D_secs[-1]**2. * L_secs[-1] + D_secs[1]**2. * 8. + D_secs[0]**2. * 4.)
		outputs['A15'] = 1025. * np.pi/4. * (D_secs[0]**2. * (-0.5) * 4.**2. - 0.5 * D_secs[1]**2. * (12.**2. - (-12.+8.)**2.) - 0.5 * D_secs[-1]**2. * ((12.+L_secs[-1])**2. - (-(12.+L_secs[-1])+L_secs[-1])**2.))
		outputs['A55'] = 1025. * np.pi/4. * (D_secs[0]**2. * 1./3.*4.**3. + D_secs[1]**2. * 8. * (1./12. * 8.**2. + (0.5*(-2.*12.+8.))**2.) + D_secs[-1]**2. * L_secs[-1] * (1./12. * L_secs[-1]**2. + (0.5*(-2.*(12.+L_secs[-1])+L_secs[-1]))**2.))