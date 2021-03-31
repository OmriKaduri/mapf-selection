import glob

from src.utils.mapfgraph import MapfGraph
from src.utils.VizMapfGraph import VizMapfGraph
from src.utils.graph_utils import from_1d_to_2d, from_2d_to_1d, sort_nodes_by_direction, find_graph_center, \
    sort_nodes_by_distance_from_center, euclidean_distance
import pandas as pd
import random
import networkx as nx


class MapfGraphGenerator(VizMapfGraph):
    def __init__(self, **args):
        VizMapfGraph.__init__(self, **args)

    def get_all_open_nodes(self):
        return [x for x, y in self.G.nodes(data=True) if y['color'] != 'white']

    def init_problem(self, instanceid):
        self.instanceid = instanceid
        self.agents_start_pos = []
        self.agents_goal_pos = []
        open_nodes = [from_1d_to_2d(n, self.grid_size[1]) for n in self.get_all_open_nodes()]
        return open_nodes

    def create_agent_from_sorted_list(self, start_sorted, reversed=False):
        start_sorted = [s for s in start_sorted if s not in self.agents_start_pos]
        goal_sorted = [g for g in start_sorted if g not in self.agents_goal_pos]
        if reversed:
            start_sorted = start_sorted[::-1]
        else:
            goal_sorted = goal_sorted[::-1]
        if len(start_sorted) == 0 or len(goal_sorted) == 0:
            return [None] * 2
        half = max(int(len(goal_sorted) * 0.5), 1)
        start = random.choice(start_sorted[:half])
        start_1d = from_2d_to_1d(start, self.grid_size[1])
        goal = random.choice(goal_sorted[:half])
        goal_1d = from_2d_to_1d(goal, self.grid_size[1])
        iterations = 0
        while not nx.has_path(self.G, start_1d, goal_1d):  # In rare cases, might happen we picked unconnected agent
            if iterations == 500:
                return [None] * 2
            start = random.choice(start_sorted[:half])
            start_1d = from_2d_to_1d(start, self.grid_size[1])
            goal = random.choice(goal_sorted[:half])
            goal_1d = from_2d_to_1d(goal, self.grid_size[1])
            iterations += 1
            print("Problem {i}".format(i=iterations))
        return start, goal

    def create_cross_side_agent(self, open_nodes, start_side, opposite_side=False, start_sorted=None):
        if not start_sorted:
            start_sorted = sort_nodes_by_direction(open_nodes, start_side)
        return self.create_agent_from_sorted_list(start_sorted, opposite_side), start_sorted

    def create_inside_out_agent(self, open_nodes, center, outside_in, start_sorted=None):
        if not start_sorted:
            start_sorted = sort_nodes_by_distance_from_center(open_nodes, center[0], center[1])
        return self.create_agent_from_sorted_list(start_sorted, outside_in), start_sorted

    def create_tight_agent(self, open_nodes, start_center, goal_center, to_tight=True,
                           start_sorted=None, goal_sorted=None):
        if not start_sorted:
            start_sorted = sort_nodes_by_distance_from_center(open_nodes, start_center[0], start_center[1])
            goal_sorted = sort_nodes_by_distance_from_center(open_nodes, goal_center[0], goal_center[1])
        start_sorted = [s for s in start_sorted if s not in self.agents_start_pos]
        goal_sorted = [g for g in goal_sorted if g not in self.agents_goal_pos]
        if len(start_sorted) == 0 or len(goal_sorted) == 0:
            return [None] * 4
        start = start_sorted[0]
        if to_tight:
            goal = goal_sorted[0]
        else:
            goal = random.choice(goal_sorted[:min(self.grid_size) * 8])
        start_1d = from_2d_to_1d(start, self.grid_size[1])
        goal_1d = from_2d_to_1d(goal, self.grid_size[1])
        iterations = 0
        while not nx.has_path(self.G, start_1d, goal_1d):  # In rare cases, might happen we picked unconnected agent
            if iterations == 500:
                return [None] * 4
            start = start_sorted[0]
            if to_tight:
                goal = goal_sorted[0]
            else:
                goal = random.choice(goal_sorted[:min(self.grid_size) * 8])
            start_1d = from_2d_to_1d(start, self.grid_size[1])
            goal_1d = from_2d_to_1d(goal, self.grid_size[1])
            iterations += 1
        return start, goal, start_sorted, goal_sorted

    # When swap_sides = True, divides agent to two groups, from starting from start_side, the other from the opposite
    def create_sided_problem(self, n_agents, start_side, instanceid, swap_sides=False):
        open_nodes = self.init_problem(instanceid)
        start_sorted = None
        for i in range(1, n_agents + 1):
            (start, goal), start_sorted = self.create_cross_side_agent(open_nodes, start_side,
                                                                       swap_sides and i % 2 == 0,
                                                                       start_sorted)
            if start is None:
                break
            self.add_agent_to_graph(start, goal)

    # When outside_in = True, creates the agents start location far from center
    def created_inside_out_problem(self, n_agents, instanceid, outside_in=False):
        open_nodes = self.init_problem(instanceid)
        center = find_graph_center(self.G, self.grid_size)
        start_sorted = None
        for i in range(1, n_agents + 1):
            (start, goal), start_sorted = self.create_inside_out_agent(open_nodes, center, outside_in, start_sorted)
            if start is None:
                break
            self.add_agent_to_graph(start, goal)

    def create_tight_problem(self, n_agents, instanceid, to_tight=True):
        open_nodes = self.init_problem(instanceid)
        start_center = random.choice(open_nodes)
        goal_center = random.choice(open_nodes)
        while euclidean_distance(start_center, goal_center) < min(self.grid_size) * 0.3:  # Be at least 0.3x aways
            goal_center = random.choice(open_nodes)
            print("TOO CLOSE")
        start_sorted = goal_sorted = None
        for i in range(1, n_agents + 1):
            start, goal, start_sorted, goal_sorted = self.create_tight_agent(open_nodes, start_center, goal_center,
                                                                             to_tight,
                                                                             start_sorted,
                                                                             goal_sorted)
            if start is None:
                break
            self.add_agent_to_graph(start, goal)

    def to_scen_file(self, path, map_name):
        with open(path, 'w+') as f:
            f.write('version 1\n')
            for agent_i, (i_start, i_goal) in enumerate(zip(self.agents_start_pos, self.agents_goal_pos)):
                f.write('1\t{m}\t{w}\t{h}\t{s_y}\t{s_x}\t{g_y}\t{g_x}\t1\n'.format(m=map_name,
                                                                                   h=self.grid_size[0],
                                                                                   w=self.grid_size[1],
                                                                                   s_y=i_start[1],
                                                                                   s_x=i_start[0],
                                                                                   g_y=i_goal[1],
                                                                                   g_x=i_goal[0]))


def create_problems_for_map(map_path, n_instances, n_agents):
    map_name = map_path.split('\\')[-1].split('.map')[0]
    problem_types = ['cross-sides', 'swap-sides', 'inside-out', 'outside-in', 'tight-to-tight', 'tight-to-wide']
    for instanceid in range(1, n_instances + 1):
        for problem_type in problem_types:
            print("Creating problem {p} of instance {i}".format(p=problem_type, i=instanceid))
            graph = MapfGraphGenerator(map_filename=map_path)
            graph.create_graph()
            additional_info = ""
            if 'sides' in problem_type:
                sides = [('L', 'R'), ('B', 'T'), ('R', 'L'), ('T', 'B')]
                start_side, goal_side = random.choice(sides)
                additional_info = "-" + start_side + '-to-' + goal_side
                if problem_type == 'cross-sides':
                    graph.create_sided_problem(n_agents, start_side, instanceid, swap_sides=False)
                elif problem_type == 'swap-sides':
                    graph.create_sided_problem(n_agents, start_side, instanceid, swap_sides=True)
            elif problem_type == 'inside-out':
                graph.created_inside_out_problem(n_agents, instanceid, outside_in=False)
            elif problem_type == 'outside-in':
                graph.created_inside_out_problem(n_agents, instanceid, outside_in=True)
            elif problem_type == 'tight-to-tight':
                graph.create_tight_problem(n_agents, instanceid, to_tight=True)
            elif problem_type == 'tight-to-wide':
                graph.create_tight_problem(n_agents, instanceid, to_tight=False)
            else:
                raise NotImplementedError("Unkown problem type {s}".format(s=problem_type))
            output_scen_path = '../../data/from-vpn/scen/custom/{m}-{pt}-{info}-{i}.scen'.format(m=map_name,
                                                                                                 pt=problem_type,
                                                                                                 i=instanceid,
                                                                                                 info=additional_info)
            image_scen_path = '../../data/from-vpn/scen/custom/{m}-{pt}-{info}-{i}.jpg'.format(m=map_name,
                                                                                               pt=problem_type,
                                                                                               i=instanceid,
                                                                                               info=additional_info)
            print("Write problem to {o}".format(o=output_scen_path))
            graph.to_scen_file(output_scen_path, map_name)
            graph.draw_graph_to(image_scen_path)


# for map_path in glob.glob('../../data/from-vpn/maps/*.map'):
#     print(map_path)
#     if 'Berlin' in map_path or 'Boston' in map_path or 'empty' in map_path or 'lak303d' in map_path \
#             or 'maze' in map_path or 'ost003d' in map_path or 'random' in map_path \
#             or 'room' in map_path:
#         continue
#     create_problems_for_map(map_path, n_instances=2, n_agents=300)
