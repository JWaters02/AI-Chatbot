# BFS algorithm in Python
import collections

# BFS algorithm
def bfs(graph, root):
    print('BFS: ')

    visited, queue = set(), collections.deque([root])
    visited.add(root)

    while queue:
        # Dequeue a vertex from queue
        vertex = queue.popleft()
        print(str(vertex) + " ", end="")

        # If not visited, mark it as visited, and
        # enqueue it
        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    print('\n')

def dfs(graph, root):
    print('DFS: ')
    visited, stack = set(), [root]
    visited.add(root)

    while stack:
        vertex = stack.pop()
        print(str(vertex) + " ", end="")

        for neighbour in graph[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    print('\n')

if __name__ == '__main__':
    graph = {0: [1, 2, 3], 1: [2], 2: [4], 3: [2], 4: [2]}
    bfs(graph, 0)
    dfs(graph, 0)
    graph = {'A': ['B', 'C'], 'B': ['F'], 'C': ['D', 'F', 'G'], 'D': ['E'], 'E': ['G'], 'F': ['G'], 'G': ['F', 'C', 'E']}
    bfs(graph, 'A')
    dfs(graph, 'A')