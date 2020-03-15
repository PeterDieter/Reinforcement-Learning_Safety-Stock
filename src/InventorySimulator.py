import numpy as np
import math
import scipy.stats as stats


class InventorySystem:

    def __init__(self, mean_daily_demand, sd_daily_demand, daily_Invcosts, daily_stockoutcosts, reorderCosts):
        self.mean_demand = mean_daily_demand
        self.sd_demand = sd_daily_demand
        self.daily_invcost = daily_Invcosts  # per unit per day
        self.daily_stockoutcost = daily_stockoutcosts  # per unit per day
        self.EOQ = round(math.sqrt((2 * self.mean_demand * reorderCosts) / daily_Invcosts))
        self.InventoryLevel = mean_daily_demand * 5  # we set the start inventory to 5 times the mean daily demand

    def init_state(self):
        # The state is just the inventory level
        return self.InventoryLevel

    def new_state(self, action):
        demand_dist = stats.truncnorm((0 - self.mean_demand) / self.sd_demand,
                                      (100 * self.sd_demand) / self.sd_demand,
                                      loc=self.mean_demand, scale=self.sd_demand)
        curr_demand = demand_dist.rvs().round().astype(int)
        self.InventoryLevel = self.InventoryLevel - curr_demand

        # if we order (action == 1) then the new inventory level is the old one plus the EOQ
        if action == 1:
            self.InventoryLevel = self.InventoryLevel + self.EOQ

        return self.InventoryLevel

    def costs(self):

        if self.InventoryLevel >= 0:  # if inventory level is positive we have inventory costs
            costs = self.InventoryLevel * self.daily_invcost
        else:  # else we have stockout costs
            costs = -self.InventoryLevel * self.daily_stockoutcost

        return costs


