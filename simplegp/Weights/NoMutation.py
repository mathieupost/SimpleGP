from gaft.plugin_interfaces.operators.mutation import Mutation


class NoMutation(Mutation):
    ''' No Mutation operator

    '''

    def __init__(self):
        pass

    def mutate(self, individual, engine):
        ''' Do no mutate the individual.

        :param individual: The individual on which crossover operation occurs
        :type individual: :obj:`gaft.components.IndividualBase`

        :param engine: Current genetic algorithm engine
        :type engine: :obj:`gaft.engine.GAEngine`

        :return: A individual
        :rtype: :obj:`gaft.components.IndividualBase`
        '''
        return individual
