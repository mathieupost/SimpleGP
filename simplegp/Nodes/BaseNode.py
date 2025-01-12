from copy import deepcopy

import numpy as np


class Node:  # Base class with general functionalities

    def __init__(self):
        self.fitness = np.inf
        self.parent = None
        self.arity = 0  # arity is the number of expected inputs
        self._children = []
        self._children_scaling = []
        self._children_translation = []

    def get_subtree(self):
        result = []
        self.__get_subtree_recursive(result)
        return result

    def get_subtree_scaling(self):
        result = []
        self.__get_subtree_scaling_recursive(result)
        return result

    def get_subtree_translation(self):
        result = []
        self.__get_subtree_translation_recursive(result)
        return result

    def set_subtree_scaling(self, scales):
        self.__set_subtree_scaling_recursive(deepcopy(scales))

    def set_subtree_translation(self, translations):
        self.__set_subtree_translation_recursive(deepcopy(translations))

    def reset_weights(self):
        number_of_weights = len(self.get_subtree_scaling())
        self.set_subtree_translation([0] * number_of_weights)
        self.set_subtree_scaling([1] * number_of_weights)

    def append_child(self, N, scale=1, translate=0):
        self._children.append(N)
        self._children_scaling.append(scale)
        self._children_translation.append(translate)
        N.parent = self

    def detach_child(self, N):
        assert (N in self._children)
        for i, c in enumerate(self._children):
            if c == N:
                self._children.pop(i)
                s = self._children_scaling.pop(i)
                t = self._children_translation.pop(i)
                N.parent = None
                return i, s, t

    def insert_child_at_position(self, i, N, scale=1, translate=0):
        self._children.insert(i, N)
        self._children_scaling.insert(i, scale)
        self._children_translation.insert(i, translate)
        N.parent = self

    def get_output(self, X):
        return None

    def get_child_output(self, i, X):
        X0 = self._children[i].get_output(X)
        X0 = deepcopy(X0)
        X0 *= self._children_scaling[i]
        X0 += self._children_translation[i]

        return X0

    def get_depth(self):
        n = self
        d = 0
        while n.parent:
            d = d + 1
            n = n.parent
        return d

    def get_subtree_readable_string(self, indent=1, weights=None):
        if weights is None:
            weights = [1, 0]
        res = f'{self}  (*{weights[0]} +{weights[1]}) \n'
        indent += 1
        for i, c in enumerate(self._children):
            res += ('\t' * indent)
            res += c.readable_string(indent, [self._children_scaling[i], self._children_translation[i]])
        return res

    def get_number_of_children(self):
        return len(self._children)

    def __get_subtree_recursive(self, result):
        result.append(self)
        for c in self._children:
            c.__get_subtree_recursive(result)
        return result

    def __get_subtree_scaling_recursive(self, result):
        for i, c in enumerate(self._children):
            result.append(self._children_scaling[i])
            c.__get_subtree_scaling_recursive(result)
        return result

    def __get_subtree_translation_recursive(self, result):
        for i, c in enumerate(self._children):
            result.append(self._children_translation[i])
            c.__get_subtree_translation_recursive(result)
        return result

    def __set_subtree_scaling_recursive(self, scales):
        for i, c in enumerate(self._children):
            self._children_scaling[i] = scales.pop(0)
            c.__set_subtree_scaling_recursive(scales)

    def __set_subtree_translation_recursive(self, translations):
        for i, c in enumerate(self._children):
            self._children_translation[i] = translations.pop(0)
            c.__set_subtree_translation_recursive(translations)
