import numpy as np

from openmdao.api import ExplicitComponent

class ECs(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		self.ECfile = EC['ECfile']

		self.add_output('windspeed_0', val=np.zeros(self.N_EC), units='m/s')
		self.add_output('Hs', val=np.zeros(self.N_EC), units='m')
		self.add_output('Tp', val=np.zeros(self.N_EC), units='s')
		self.add_output('p', val=np.zeros(self.N_EC))

	def compute(self,inputs,outputs):
		outputs['windspeed_0'], outputs['Hs'], outputs['Tp'], outputs['p'] = np.loadtxt(self.ECfile, unpack=True)