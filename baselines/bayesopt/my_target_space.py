import bayes_opt.target_space import TargetSpace


class MyTargetSpace(TargetSpace):
    def __init__(self, target_func, alleles, max_len, random_state=None):
        self.random_state = ensure_rng(random_state)
        
        self.target_func = target_func

        self.alleles = alleles
        
        self._params = []
        self._target = np.empty(shape=(0))

        self._cache = {}

    def probe(self, param):
        try:
            target = self._cache[param]

        except KeyError:
            target = self.target_func(peptides=[param[0]], alleles=[param[1]])

            self.register(param, target)

        return target

    def register(self, param, target):
        if param in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[param] = target

        self._params.append(param)
        self._target = np.concatenate([self._target, [target])

        


