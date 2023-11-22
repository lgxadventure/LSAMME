import numpy as np
import numbers
import copy

def clone(estimator):
    estimator_type = type(estimator)
    if estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e) for e in estimator])
    elif not hasattr(estimator, "get_params") or isinstance(estimator, type):
        return copy.deepcopy(estimator)

    estimator_class = estimator.__class__
    new_params = estimator.get_params(deep=False)
    for name, param in new_params.items():
        new_params[name] = clone(param)

    new_object = estimator_class(**new_params)

    return new_object

def _check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState instance" % seed)

def _set_random_states(estimator, random_state):
    random_state = _check_random_state(random_state)
    to_set = {}
    for key in sorted(estimator.get_params(deep=True)):
        if key == "random_state" or key.endswith("__random_state"):
            to_set[key] = random_state.randint(np.iinfo(np.int32).max)
    
    if to_set:
        estimator.set_params(**to_set)