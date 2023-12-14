from typing import Text, Any, Iterable


class Trie:
    def __init__(self):
        self.root = {}
    
    def add(self, edges: Iterable):
        node = self.root
        for edge in edges:
            if edge not in node:
                node[edge] = {}
            node = node[edge]
        node["<END>"] = None
    
    def exists(self, edges: Iterable):
        node = self.root
        for edge in edges:
            if edge not in node:
                return False
            node = node[edge]
        if "<END>" not in node:
            return False
        return True

    def get_entities_by_prefix(self, prefix: Iterable):
        node = self.root
        for edge in prefix:
            if edge not in node:
                return []
            node = node[edge]
        L = self._get_entities(node)
        L = [(*prefix, *entity) for entity in L]
        return L

    @staticmethod
    def _get_entities(obj: dict):
        L = []
        stack = [((), obj)]
        while stack:
            entity, node = stack.pop()
            for k, v in node.items():
                if k == "<END>":
                    L.append(entity)
                else:
                    stack.append(((*entity, k), v))
        return L
