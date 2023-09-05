import netgen

class NetgenConfig:
    def __init__(self):
        self.seed = -1
        self.num_nodes = 11
        self.num_arcs = 30
        self.num_provisioned_bundles = 0
        self.num_all_bundles = 16
        self.num_priorities = 10
        self.bandwidth_lb = 15_000
        self.bandwidth_ub = 250_000
        self.latency_lb = 295_000
        self.latency_ub = 600_000
        self.rate_lb = 128
        self.rate2_lb = 128
        self.rate_ub = 65_536
        self.rate2_ub = 65_536
        self.delay_lb = 1_080_000
        self.delay2_lb = 1_080_000
        self.delay_ub = 9_072_000
        self.delay2_ub = 9_072_000
        self.num_unicast_reqs_per_bundle_lb = 0
        self.num_unicast_reqs_per_bundle_ub = 1
        self.bundle_size_lb = 1
        self.bundle_size_ub = 1
        self.request_size_lb = 2
        self.request_size_ub = 5

config = NetgenConfig()


from graph_envs import multicast_onr

print('In test!')
