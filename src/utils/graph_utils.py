import sys
import networkx as nx
import igraph as ig


def left_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), (row_index * width + cell_index - 1)
    else:
        return (row_index * width + cell_index - 1), (row_index * width + cell_index)


def right_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), (row_index * width + cell_index + 1)
    else:
        return (row_index * width + cell_index + 1), (row_index * width + cell_index)


def bottom_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), ((row_index + 1) * width + cell_index)
    else:
        return ((row_index + 1) * width + cell_index), (row_index * width + cell_index)


def top_edge(row_index, width, cell_index, direction='out'):
    if direction == 'out':
        return (row_index * width + cell_index), ((row_index - 1) * width + cell_index)
    else:
        return ((row_index - 1) * width + cell_index), (row_index * width + cell_index)


def mark_cell_as_obstacle(cell_index, row_index, grid_size, graph):
    graph.add_node(row_index * grid_size[1] + cell_index, color='white', type='obstacle', size=1)
    try:
        if graph.has_edge(*left_edge(row_index, grid_size[1], cell_index, direction='in')):  # Remove edge from left
            graph.remove_edge(*left_edge(row_index, grid_size[1], cell_index, direction='in'))
        if graph.has_edge(*top_edge(row_index, grid_size[1], cell_index, direction='in')):  # Remove edge from up
            graph.remove_edge(*top_edge(row_index, grid_size[1], cell_index, direction='in'))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        print(cell_index, row_index)
        print("Tried to remove an edge already removed")


def mark_cell_as_free(cell_index, row_index, grid_size, graph, free_index=-1):
    if free_index != -1:
        graph.add_node(row_index * grid_size[1] + cell_index, color='black', type='open', size=1, free_index=free_index)
    else:
        graph.add_node(row_index * grid_size[1] + cell_index, color='black', type='open', size=1)

    if cell_index > 0 and graph.has_edge(
            *left_edge(row_index, grid_size[1], cell_index, direction='in')):  # Create Edge to left
        graph.add_edge(*left_edge(row_index, grid_size[1], cell_index), weight=3)
    if cell_index < grid_size[1] - 1:  # Create Edge to Right
        graph.add_edge(*right_edge(row_index, grid_size[1], cell_index), weight=3)
    if row_index < grid_size[0] - 1:  # Create Edge to Bottom
        graph.add_edge(*bottom_edge(row_index, grid_size[1], cell_index), weight=3)
    if row_index > 0 and graph.has_edge(
            *top_edge(row_index, grid_size[1], cell_index, direction='in')):  # Create Edge to Top
        graph.add_edge(*top_edge(row_index, grid_size[1], cell_index), weight=3)


def agent_points_from(metadata, grid_width):
    goal_x = int(metadata[1])
    goal_y = int(metadata[2])
    start_x = int(metadata[3])
    start_y = int(metadata[4])

    return start_x * grid_width + start_y, goal_x * grid_width + goal_y


def from_2d_to_1d(point, grid_width):
    return point[0] * grid_width + point[1]


def from_1d_to_2d(point, grid_width):
    return point // grid_width, point % grid_width


def rotate_positions_90_clockwise(x, y):
    return y, -x


def networkx_to_igraph(graph):
    return ig.Graph(len(graph), list(zip(*list(zip(*nx.to_edgelist(graph)))[:2])))


def euclidean_distance(a, b):
    return 0.5 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def sort_nodes_by_distance_from_center(nodes, center_x, center_y):
    sorted_nodes = sorted(nodes, key=lambda xy: euclidean_distance(xy, [center_x, center_y]))
    return sorted_nodes


def find_graph_center(G, grid_size):
    center_h, center_x = grid_size[0], grid_size[1]
    center = from_2d_to_1d((center_h // 2, center_x // 2), grid_size[1])
    center_found = G.nodes[center]['color'] != 'white'
    while not center_found:
        center = center + 1
        center_found = G.nodes[center]['color'] != 'white'
    return from_1d_to_2d(center, grid_size[1])


def sort_nodes_by_direction(nodes, direction):
    sorted_nodes = []
    if direction == 'L':
        sorted_nodes = sorted(nodes, key=lambda xy: xy[1])  # Smaller X-vals first
    elif direction == 'R':
        sorted_nodes = sorted(nodes, key=lambda xy: xy[1], reverse=True)  # Bigger X-vals first
    elif direction == 'T':
        sorted_nodes = sorted(nodes, key=lambda xy: xy[0])  # Smaller Y-vals first
    elif direction == 'B':
        sorted_nodes = sorted(nodes, key=lambda xy: xy[0], reverse=True)  # Bigger Y-vals first

    return sorted_nodes
