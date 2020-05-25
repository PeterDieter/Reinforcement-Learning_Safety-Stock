import numpy as np
import math
import scipy.stats as stats


class InventorySystem:

    def __init__(self, mean_daily_demand, sd_daily_demand, daily_invcosts, daily_stockoutcosts, reorderCosts):
        self.mean_demand = mean_daily_demand
        self.sd_demand = sd_daily_demand
        self.daily_invcost = daily_invcosts  # per unit per day
        self.daily_stockoutcost = daily_stockoutcosts  # per unit per day
        self.eoq = round(math.sqrt((2 * self.mean_demand * reorderCosts) / daily_invcosts))
        self.inventoryLevel = mean_daily_demand * 5  # we set the start inventory to 5 times the mean daily demand

    def init_state(self):
        # The state is just the inventory level
        self.inventoryLevel = self.mean_demand * 3
        return np.array([self.inventoryLevel])

    def new_state(self, action):
        # demand is normal distributed
        demand_dist = stats.truncnorm((0 - self.mean_demand) / (self.sd_demand + 0.0001),
                                      (100 * self.sd_demand) / (self.sd_demand + 0.0001),
                                      loc=self.mean_demand, scale=self.sd_demand)
        curr_demand = demand_dist.rvs().round().astype(int)
        self.inventoryLevel = self.inventoryLevel - curr_demand

        # if we order (action == order) then the new inventory level is the old one plus the EOQ
        if action == "order":
            self.inventoryLevel = self.inventoryLevel + self.eoq

        return np.array([self.inventoryLevel])

    def costs(self):

        if self.inventoryLevel >= 0:  # if inventory level is positive we have inventory costs
            costs = self.inventoryLevel * self.daily_invcost
        else:  # else we have stockout costs
            costs = -self.inventoryLevel * self.daily_stockoutcost

        return costs


