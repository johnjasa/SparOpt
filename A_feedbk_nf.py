import numpy as np

from openmdao.api import ExplicitComponent

class Afeedbk(ExplicitComponent):

	def setup(self):
		self.add_input('A_struct', val=np.zeros((7,7)))
		self.add_input('A_contrl', val=np.zeros((4,4)))
		self.add_input('BsCc', val=np.zeros((7,4)))
		self.add_input('BcCs', val=np.zeros((4,7)))

		self.add_output('A_feedbk', val=np.zeros((11,11)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['A_feedbk'] = np.concatenate((np.concatenate((inputs['A_struct'],inputs['BsCc']),1),np.concatenate((inputs['BcCs'],inputs['A_contrl']),1)),0)

	def compute_partials(self, inputs, partials):
		As_arr1 = np.concatenate((np.identity(7),np.zeros((4,7))),0)
		BsCc_arr1 = np.concatenate((np.zeros((7,4)),np.identity(4)),0)
		As_arr2 = np.concatenate((As_arr1,np.zeros((11,42))),1)
		BsCc_arr2 = np.concatenate((BsCc_arr1,np.zeros((11,24))),1)
		for i in xrange(1,6):
			As_arr2 = np.concatenate((As_arr2,np.concatenate((np.zeros((11,7*i)),As_arr1,np.zeros((11,7*(6-i)))),1)),0)
			BsCc_arr2 = np.concatenate((BsCc_arr2,np.concatenate((np.zeros((11,4*i)),BsCc_arr1,np.zeros((11,4*(6-i)))),1)),0)
		
		As_arr2 = np.concatenate((As_arr2,np.concatenate((np.zeros((11,42)),As_arr1),1)),0)
		BsCc_arr2 = np.concatenate((BsCc_arr2,np.concatenate((np.zeros((11,24)),BsCc_arr1),1)),0)

		partials['A_feedbk', 'A_struct'] = np.concatenate((As_arr2,np.zeros((44,49))),0)
		partials['A_feedbk', 'A_contrl'] = np.concatenate((np.concatenate((np.zeros((84,4)),np.identity(4),np.zeros((33,4))),0),np.concatenate((np.zeros((95,4)),np.identity(4),np.zeros((22,4))),0),np.concatenate((np.zeros((106,4)),np.identity(4),np.zeros((11,4))),0),np.concatenate((np.zeros((117,4)),np.identity(4)),0)),1)
		partials['A_feedbk', 'BsCc'] = np.concatenate((BsCc_arr2,np.zeros((44,28))),0)
		partials['A_feedbk', 'BcCs'] = np.concatenate((np.concatenate((np.zeros((77,7)),np.identity(7),np.zeros((37,7))),0),np.concatenate((np.zeros((88,7)),np.identity(7),np.zeros((26,7))),0),np.concatenate((np.zeros((99,7)),np.identity(7),np.zeros((15,7))),0),np.concatenate((np.zeros((110,7)),np.identity(7),np.zeros((4,7))),0)),1)