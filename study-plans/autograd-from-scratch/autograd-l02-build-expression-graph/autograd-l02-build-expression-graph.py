import numpy as np

def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """

    nodes = []

    # Create leaf nodes
    for leaf in leaves:
        node = {
            'id': leaf['id'],
            'data': leaf['data'],
            'grad': 0.0,
            'op': '',
            'parents': []
        }

        nodes.append(node)

    # Map ID -> node
    node_map = {node['id']: node for node in nodes}

    # Create operation nodes
    for operation in operations:
        left = node_map[operation['left']]
        right = node_map[operation['right']]

        if operation['op'] == '+':
            data = left['data'] + right['data']
        elif operation['op'] == '*':
            data = left['data'] * right['data']
        else:
            raise ValueError(f"Unknown operation: {operation['op']}")

        node = {
            'id': operation['id'],
            'data': data,
            'grad': 0.0,
            'op': operation['op'],
            'parents': [
                operation['left'],
                operation['right']
            ]
        }

        nodes.append(node)
        node_map[node['id']] = node

    # If there are operations, the last operation is the output.
    # Otherwise, the last leaf is the output.
    if operations:
        final_id = operations[-1]['id']
    else:
        final_id = leaves[-1]['id']

    return nodes, final_id