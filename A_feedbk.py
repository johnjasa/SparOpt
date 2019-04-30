import numpy as np

from openmdao.api import ExplicitComponent

class Afeedbk(ExplicitComponent):

	def setup(self):
		self.add_input('A_struct', val=np.zeros((7,7)))
		self.add_input('A_contrl', val=np.zeros((2,2)))
		self.add_input('BsCc', val=np.zeros((7,2)))
		self.add_input('BcCs', val=np.zeros((2,7)))

		self.add_output('A_feedbk', val=np.zeros((9,9)))

		self.declare_partials('*', '*')

		# I'm not sure what's going on in the compute_partials (looks like some complicated array stuff),
		# but I think this could definitely benefit from sparsity in the declare_partials.

	def compute(self, inputs, outputs):
		outputs['A_feedbk'] = np.concatenate((np.concatenate((inputs['A_struct'],inputs['BsCc']),1),np.concatenate((inputs['BcCs'],inputs['A_contrl']),1)),0)

	def compute_partials(self, inputs, partials):
		As_arr1 = np.concatenate((np.identity(7),np.zeros((2,7))),0)
		BsCc_arr1 = np.concatenate((np.zeros((7,2)),np.identity(2)),0)
		As_arr2 = np.concatenate((As_arr1,np.zeros((9,42))),1)
		BsCc_arr2 = np.concatenate((BsCc_arr1,np.zeros((9,12))),1)
		for i in xrange(1,6):
			As_arr2 = np.concatenate((As_arr2,np.concatenate((np.zeros((9,7*i)),As_arr1,np.zeros((9,7*(6-i)))),1)),0)
			BsCc_arr2 = np.concatenate((BsCc_arr2,np.concatenate((np.zeros((9,2*i)),BsCc_arr1,np.zeros((9,2*(6-i)))),1)),0)

		As_arr2 = np.concatenate((As_arr2,np.concatenate((np.zeros((9,42)),As_arr1),1)),0)
		BsCc_arr2 = np.concatenate((BsCc_arr2,np.concatenate((np.zeros((9,12)),BsCc_arr1),1)),0)

		partials['A_feedbk', 'A_struct'] = np.concatenate((As_arr2,np.zeros((18,49))),0)
		partials['A_feedbk', 'A_contrl'] = np.concatenate((np.concatenate((np.zeros((70,2)),np.identity(2),np.zeros((9,2))),0),np.concatenate((np.zeros((79,2)),np.identity(2)),0)),1)
		partials['A_feedbk', 'BsCc'] = np.concatenate((BsCc_arr2,np.zeros((18,14))),0)
		partials['A_feedbk', 'BcCs'] = np.concatenate((np.concatenate((np.zeros((63,7)),np.identity(7),np.zeros((11,7))),0),np.concatenate((np.zeros((72,7)),np.identity(7),np.zeros((2,7))),0)),1)
