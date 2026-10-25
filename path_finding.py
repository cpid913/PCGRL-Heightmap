import heapq

'''
The player can only climb one block high or jump one block below
'''
def can_move(heightmap, a, b) -> bool:
    return abs(heightmap[b[0]][b[1]] - heightmap[a[0]][a[1]]) <= 1

'''
Return a list of reachable neighbors from the current cell
'''
def get_neighbors(heightmap, cell):
    neighbors = []
    rows, cols = len(heightmap), len(heightmap[0])
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and can_move(heightmap, cell, (nr, nc)):
            neighbors.append((nr, nc))
    return neighbors

'''
Manhattan distance as the heuristic function
'''
def heuristic(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

'''
A*
'''
def Astar(heightmap, start, goal):
    # The search heap that pops the cell with the smallest heuristic score
    search_heap = [(heuristic(start, goal), 0, start)]

    # For constructing the path
    came_from = {}

    # This dictionary stores the path length between each cell and the start point
    g_dict = {start: 0}

    visited = set()

    while search_heap:
        _, g, current = heapq.heappop(search_heap)
        # Skip visited cell
        if current in visited:
            continue
        # Return the path if goal is reached
        if current == goal:
            visited.add(current)
            return reconstruct(came_from, current), visited

        visited.add(current)

        for nb in get_neighbors(heightmap, current):
            # Skip visited cell
            if nb in visited:
                continue
            next_g = g + 1  # uniform step cost
            if next_g < g_dict.get(nb, float("inf")):
                g_dict[nb] = next_g
                came_from[nb] = current
                f = next_g + heuristic(nb, goal)
                heapq.heappush(search_heap, (f, next_g, nb))

    return None

'''
Get the path
'''
def reconstruct(came_from, current):
    if current not in came_from:
        return [current]
    return reconstruct(came_from, came_from[current]) + [current]