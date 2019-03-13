import numpy as np

from openmdao.api import Group

from hull_r_hull import HullRHull
from hull_Z_l import HullZL
from hull_I_C import HullIC
from hull_A_C import HullAC
from hull_gyr_radius import HullGyrRadius
from ring_buckling_1 import RingBuckling1
from ring_buckling_2 import RingBuckling2
from col_buckling import ColBuckling
from constr_area_ringstiff import ConstrAreaRingstiff

class HullBuckling(Group):

	def setup(self):
		
		self.add_subsystem('hull_r_hull', HullRHull(), promotes_inputs=['D_spar_p', 'wt_spar_p'], promotes_outputs=['r_hull'])

		self.add_subsystem('hull_Z_l', HullZL(), promotes_inputs=['l_stiff', 'r_hull', 'wt_spar_p'], promotes_outputs=['Z_l'])
		
		self.add_subsystem('hull_I_C', HullIC(), promotes_inputs=['D_spar_p', 'wt_spar_p'], promotes_outputs=['I_C'])
		
		self.add_subsystem('hull_A_C', HullAC(), promotes_inputs=['r_hull', 'wt_spar_p'], promotes_outputs=['A_C'])
		
		self.add_subsystem('hull_gyr_radius', HullGyrRadius(), promotes_inputs=['I_C', 'A_C'], promotes_outputs=['i_C'])
		
		self.add_subsystem('ring_buckling_1', RingBuckling1(), promotes_inputs=['h_stiff', 't_w_stiff', 'f_y'], promotes_outputs=['ring_buckling_1'])
		
		self.add_subsystem('ring_buckling_2', RingBuckling2(), promotes_inputs=['h_stiff', 'b_stiff', 'r_hull', 'f_y'], promotes_outputs=['ring_buckling_2'])
		
		self.add_subsystem('col_buckling', ColBuckling(), promotes_inputs=['buck_len', 'spar_draft', 'i_C', 'f_y'], promotes_outputs=['col_buckling'])
		
		self.add_subsystem('constr_area_ringstiff', ConstrAreaRingstiff(), promotes_inputs=['A_R', 'Z_l', 'l_stiff', 'wt_spar_p'], promotes_outputs=['constr_area_ringstiff'])