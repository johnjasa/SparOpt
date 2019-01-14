import numpy as np

from openmdao.api import Group

from spar_cost import SparCost
from tower_cost import TowerCost
from mooring_cost import MooringCost
from total_cost import TotalCost

class Cost(Group):

	def setup(self):
	 	self.add_subsystem('spar_cost', SparCost(), promotes_inputs=['D_spar', 'D_spar_p', 'wt_spar', 'L_spar', 'l_stiff', 'h_stiff', 't_f_stiff', 'A_R', 'r_f', 'r_e', 'tot_M_spar'], promotes_outputs=['spar_cost'])

	 	self.add_subsystem('tower_cost', TowerCost(), promotes_inputs=['D_tower', 'D_tower_p', 'wt_tower', 'L_tower', 'tot_M_tower'], promotes_outputs=['tower_cost'])

	 	self.add_subsystem('mooring_cost', MooringCost(), promotes_inputs=['len_tot_moor', 'mass_dens_moor'], promotes_outputs=['mooring_cost'])

	 	self.add_subsystem('total_cost', TotalCost(), promotes_inputs=['spar_cost', 'tower_cost', 'mooring_cost'], promotes_outputs=['total_cost'])