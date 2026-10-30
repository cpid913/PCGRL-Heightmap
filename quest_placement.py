import numpy as np

MIN_QUEST_GAP = 4       # Straight line distance in cells between any two quest points
MAX_SIDE_QUESTS = 5     # Maximum number of side quests to place
MAX_ATTEMPTS = 30       # Maximum number of attempts to place a quest

'''
Straight line distance between two cells
'''
def cell_distance(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

'''
Poisson disk sampling over the whole map, terrain is ignored here.
Reachability is checked later by floodfill
'''
def place_quests(heightmap, rng):
    rows, cols = heightmap.shape
    side_count = int(rng.integers(0, MAX_SIDE_QUESTS + 1))
    # Spawn, main quest, then the side quests
    wanted = 2 + side_count
    placed = []

    while len(placed) < wanted:
        found = False
        for _ in range(MAX_ATTEMPTS):
            cell = (int(rng.integers(rows)), int(rng.integers(cols)))
            # Reject the cell if it is too close to any placed point
            if all(cell_distance(cell, other) >= MIN_QUEST_GAP for other in placed):
                placed.append(cell)
                found = True
                break
        # Map is full sostop with what we have
        if not found:
            break

    spawn, main_quest, side_quests = placed[0], placed[1], placed[2:]
    print(f'Side quests: {len(side_quests)} placed out of {side_count} wanted')
    return spawn, main_quest, side_quests
