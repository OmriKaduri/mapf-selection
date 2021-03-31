import linecache
import os
import zipfile
import itertools

from src.utils.graph_utils import mark_cell_as_free, mark_cell_as_obstacle, from_1d_to_2d, \
    from_2d_to_1d, networkx_to_igraph
import numpy as np

import networkx as nx
from timeit import default_timer as timer
import seaborn as sns
from matplotlib.colors import to_rgb


def find_k_collisions(path, paths, k):
    k_collisions_num = 0
    for k_paths in itertools.combinations(paths, k - 1):
        k_paths = (*k_paths, path)
        max_length = min([len(path) for path in k_paths])  # No k-way collisions can be found with less than k agents
        k_paths = [path[:max_length] for path in k_paths]
        k_collisions_num += ((np.diff(np.vstack(k_paths), axis=0) == 0).sum(axis=0) == k - 1).sum()

    return k_collisions_num


def grid_name_from_map(map):
    if '/' in map:
        sep = '/'
    else:
        sep = '\\'
    return map.split('.map')[0].split(sep)[-1]


def get_cmap(n, red_green=False):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard seaborn colormap name.'''
    if red_green:
        return ['green', 'red'] * int(n)
    return sns.color_palette("viridis", n)


class MapfGraph:
    def __init__(self, map_filename):
        self.map_filename = map_filename
        self.instanceid = 0
        self.grid_size = [-1, -1]
        self.num_agents = -1
        self.num_obstacles = 0
        self.G = nx.empty_graph()
        self.agents_start_pos = []
        self.agents_goal_pos = []
        self.agent_locations = []
        self.features = {}
        self.cells = {}
        self.extracted_for_agents = 0
        self.paths = []
        self.path_lengths = []
        self.start_to_start_paths = []
        self.s2s_path_lengths = []
        self.goal_to_goal_paths = []
        self.g2g_path_lengths = []
        self.num_k_collisions = {2: 0}
        self.agents_done = {}
        self.igraph_G = None

    def create_graph(self):
        with open(self.map_filename) as f:
            for index, line in enumerate(f):
                if index == 1:
                    height = int(line.split(' ')[1])
                    # Done for cases when the instance is of the format X,MORETEXT => X
                elif index == 2:
                    width = int(line.split(' ')[1])
                    self.grid_size = [height, width]
                    self.G = nx.DiGraph()
                elif 3 < index < self.grid_size[0] + 4:
                    # Do for all lines representing the grid. grid_size[1] is the grid width
                    for cell_index, cell in enumerate(line):
                        if cell == '.':
                            mark_cell_as_free(cell_index, index - 4, self.grid_size, self.G)
                        elif cell == '@' or cell == 'T':
                            mark_cell_as_obstacle(cell_index, index - 4, self.grid_size, self.G)
                            self.num_obstacles += 1

                elif index == self.grid_size[0] + 4:  # Number of agents line
                    self.num_agents = int(line)

                # elif index > self.grid_size[0] + 4:
                # agent_data = line.split(',')
                # start, goal = agent_points_from(agent_data, self.grid_size[1])
                # self.agent_locations.append((start, goal))
                # self.G.add_node(goal, color="green",type='agent_goal', size=1)
                # self.G.add_node(start, color="red", size=1)

        self.igraph_G = networkx_to_igraph(self.G)

    @staticmethod
    def agents_from_scen_row(agent_row):
        agent_data = agent_row.split()
        start_y = int(agent_data[4])
        start_x = int(agent_data[5])
        goal_y = int(agent_data[6])
        goal_x = int(agent_data[7])
        return (start_x, start_y), (goal_x, goal_y)

    def load_agents_from_scen(self, scenfile, n_agents, instanceid, n_agent_start=0):
        red_green=True
        self.instanceid = instanceid
        self.agents_start_pos = []
        self.agents_goal_pos = []
        self.num_agents = n_agents
        cmap = get_cmap(n_agents, red_green=red_green)
        for i in range(n_agent_start + 2, n_agent_start + n_agents + 2):
            agent_row = linecache.getline(scenfile, i)
            if agent_row == '':  # No such agent exists
                continue
            start, goal = MapfGraph.agents_from_scen_row(agent_row)
            if red_green:
                self.add_agent_to_graph(start, goal, color=None)
            else:
                self.add_agent_to_graph(start, goal, color=cmap[i-2])

    def add_agent_to_graph(self, start, goal, color=None):
        if color is None:
            start_color = to_rgb('green')
            goal_color = to_rgb('red')
        else:
            if len(color) == 2:
                start_color = color[0]
                goal_color = color[1]
            else:
                start_color = goal_color = color
        self.agents_start_pos.append(start)
        self.agents_goal_pos.append(goal)
        start_1d = from_2d_to_1d(start, self.grid_size[1])
        goal_1d = from_2d_to_1d(goal, self.grid_size[1])

        self.G.nodes[start_1d]['color'] = start_color
        self.G.nodes[start_1d]['size'] = 1
        self.G.nodes[start_1d]['type'] = 'agent_start'

        self.G.node[goal_1d]['color'] = goal_color
        self.G.node[goal_1d]['size'] = 1
        self.G.nodes[goal_1d]['type'] = 'agent_goal'

    def feature_extraction(self, with_cell_features=True):
        fit_start = timer()
        if len(self.agents_start_pos) == 2:  # Init cells features
            self.init_cells()

        for agent_i, (i_start, i_goal) in enumerate(zip(self.agents_start_pos, self.agents_goal_pos)):
            if agent_i < self.extracted_for_agents:
                continue
            start_cell = self.get_node_cell(i_start)
            goal_cell = self.get_node_cell(i_goal)
            if with_cell_features:
                self.cells[f'{start_cell[0]}_{start_cell[1]}_cell_agent_start'] += 1
                self.cells[f'{goal_cell[0]}_{goal_cell[1]}_cell_agent_goal'] += 1
            start = from_2d_to_1d(i_start, self.grid_size[1])
            goal = from_2d_to_1d(i_goal, self.grid_size[1])

            path = self.igraph_G.get_shortest_paths(v=start, to=goal)[0]
            for k, n_collisions in self.num_k_collisions.items():
                self.num_k_collisions[k] += find_k_collisions(path, self.paths, k)
            self.paths.append(path)
            self.path_lengths.append(len(path) - 1)  # Minus 1 due the path starts from the node
            j_start_points = []
            j_goal_points = []
            for agent_j, (j_start, j_goal) in enumerate(zip(self.agents_start_pos, self.agents_goal_pos)):
                if (agent_j == agent_i) or (str(agent_i) + '_' + str(agent_j) in self.agents_done):
                    continue
                j_start_points.append(from_2d_to_1d(j_start, self.grid_size[1]))
                j_goal_points.append(from_2d_to_1d(j_goal, self.grid_size[1]))
                self.agents_done[str(agent_i) + '_' + str(agent_j)] = 1
                self.agents_done[str(agent_j) + '_' + str(agent_i)] = 1

            s2s_paths = self.igraph_G.get_shortest_paths(v=start, to=j_start_points)
            g2g_paths = self.igraph_G.get_shortest_paths(v=goal, to=j_goal_points)

            # self.start_to_start_paths.append(s2s_path)
            self.s2s_path_lengths.extend([len(p) - 1 for p in s2s_paths])
            # self.goal_to_goal_paths.append(g2g_path)
            self.g2g_path_lengths.extend([len(p) - 1 for p in g2g_paths])

            self.extracted_for_agents += 1

        all_paths = np.concatenate(self.paths)
        np_path_lengths = np.array(self.path_lengths)
        # start_to_start_paths = np.array(self.start_to_start_paths)
        np_s2s_path_lengths = np.array(self.s2s_path_lengths)
        # goal_to_goal_paths = np.array(self.goal_to_goal_paths)
        np_g2g_path_lengths = np.array(self.g2g_path_lengths)

        total_grid_size = self.grid_size[0] * self.grid_size[1]
        self.features['GridName'] = grid_name_from_map(self.map_filename)
        self.features['GridRows'] = int(self.grid_size[0])
        self.features['GridColumns'] = int(self.grid_size[1])
        self.features['NumOfAgents'] = int(self.num_agents)
        self.features['NumOfObstacles'] = int(self.num_obstacles)
        self.features['InstanceId'] = self.instanceid
        # self.features['BranchingFactor'] = 5 ** self.num_agents  # Assuming only 5-moves grid MAPF
        self.features['ObstacleDensity'] = self.num_obstacles / total_grid_size
        self.features['AvgDistanceToGoal'] = np_path_lengths.mean()
        self.features['MaxDistanceToGoal'] = np_path_lengths.max()
        self.features['MinDistanceToGoal'] = np_path_lengths.min()
        self.features['StdDistanceToGoal'] = np_path_lengths.std()
        self.features['AvgStartDistances'] = np_s2s_path_lengths.mean()
        self.features['AvgGoalDistances'] = np_g2g_path_lengths.mean()
        self.features['MaxStartDistances'] = np_s2s_path_lengths.max()
        self.features['MaxGoalDistances'] = np_g2g_path_lengths.max()
        self.features['MinStartDistances'] = np_s2s_path_lengths.min()
        self.features['MinGoalDistances'] = np_g2g_path_lengths.min()
        self.features['StdStartDistances'] = np_s2s_path_lengths.std()
        self.features['StdGoalDistances'] = np_g2g_path_lengths.std()

        unique_points_on_shortest_paths = set(all_paths.ravel())
        self.features['PointsAtSPRatio'] = len(unique_points_on_shortest_paths) / total_grid_size
        self.features['Sparsity'] = self.num_agents / (total_grid_size - self.num_obstacles)

        # TODO: Ideas for features - PointsAtSPRatio computes the ratio between unique number of nodes on shortest paths to
        #                            total number of nodes. We need a feature that describes how "serious" the collisions of the SPs.
        #                            I.e. - If a collision in paths is made between 2 agents and 5 agents, it's should be noted in the features
        #                            Naive idea: Add features of 2-way collisions, 3-way untill 10+way, and in each one count the number of collisions.

        self.features['2waycollisions'] = self.num_k_collisions[2]
        # self.features['3waycollisions'] = self.num_k_collisions[3]
        # self.features['4waycollisions'] = self.num_k_collisions[4]

        # TODO: Can we somehow extract "locality" features? i.e. How many "crowded" regions are there?
        if with_cell_features:
            for i in range(0, 8):
                for j in range(0, 8):
                    self.features[f'{i}_{j}_cell_agent_start'] = self.cells[
                                                                     f'{i}_{j}_cell_agent_start'] / self.num_agents
                    self.features[f'{i}_{j}_cell_agent_goal'] = self.cells[f'{i}_{j}_cell_agent_goal'] / self.num_agents
                    self.features[f'{i}_{j}_cell_obstacles'] = self.cells[f'{i}_{j}_cell_obstacles']
                    self.features[f'{i}_{j}_cell_open'] = self.cells[f'{i}_{j}_cell_open']

    def init_cells(self):
        n_open_cells = self.grid_size[0] * self.grid_size[1] - self.num_obstacles

        for i in range(0, self.grid_size[0], self.grid_size[0] // 8):
            for j in range(0, self.grid_size[1], self.grid_size[1] // 8):
                cell = (i, j)
                cell_pos = (i // (self.grid_size[0] // 8), j // (self.grid_size[1] // 8))
                relevant_nodes = [(node, features) for node, features in self.G.nodes(data=True) if
                                  self.in_cell_range(node, cell)]
                open_ratio = len(
                    [n for n in relevant_nodes if n[1]['type'] == 'open'])
                if self.num_obstacles == 0:
                    obstacle_ratio = 0
                else:
                    obstacle_ratio = len(
                        [n for n in relevant_nodes if n[1]['type'] == 'obstacle']) / self.num_obstacles

                self.cells[f'{cell_pos[0]}_{cell_pos[1]}_cell_agent_start'] = 0
                self.cells[f'{cell_pos[0]}_{cell_pos[1]}_cell_agent_goal'] = 0
                self.cells[f'{cell_pos[0]}_{cell_pos[1]}_cell_obstacles'] = obstacle_ratio
                self.cells[f'{cell_pos[0]}_{cell_pos[1]}_cell_open'] = open_ratio / n_open_cells

    def in_cell_range(self, node, cell):
        node_2d = from_1d_to_2d(node, self.grid_size[1])
        if cell[0] <= node_2d[0] < min(self.grid_size[0], cell[0] + self.grid_size[0] // 8) and \
                cell[1] <= node_2d[1] < min(self.grid_size[1], cell[1] + self.grid_size[1] // 8):
            return True

        return False

    def get_node_cell(self, node):
        row_jump = self.grid_size[0] // 8
        col_jump = self.grid_size[1] // 8
        return (node[0] // row_jump), (node[1] // col_jump)


def saveCompressed(fh, **namedict):
    with zipfile.ZipFile(fh, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for k, v in namedict.items():
            with zf.open(k + '.npy', 'w', force_zip64=True) as buf:
                np.lib.npyio.format.write_array(buf,
                                                np.asanyarray(v),
                                                allow_pickle=False)

# map_path = '../../data/from-vpn/maps/Berlin_1_256.map'
# scen_path = '../../data/from-vpn/scen/scen1/Berlin_1_256-even-3.scen'
#
# n_agents = 10
# graph = MapfGraph(map_path)
# graph.create_graph()
# graph.load_agents_from_scen(scen_path, n_agents, 3)
# graph.feature_extraction()
