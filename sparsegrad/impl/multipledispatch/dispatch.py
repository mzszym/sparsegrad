__all__ = [ 'Dispatcher' ]

def supersedes(a, b):
    """a is strictly more specific than b"""
    if len(a) != len(b):
        return False
    return all(map(issubclass,a,b))

def _supersedes(a,b):
    return all(map(issubclass,a,b))

class DispatchError(ValueError):
    pass

def super_signature(signatures):
    """ A signature that would break ambiguities """
    n = len(signatures[0])
    assert all(len(s) == n for s in signatures)

    return [max([type.mro(sig[i]) for sig in signatures], key=len)[0]
            for i in range(n)]

class RaiseDispatchError(object):
    def __init__(self, signature, candidates):
        self.signature = signature
        self.candidates = candidates

    def __call__(self, *args, **kwargs):
        raise DispatchError('Cannot unambigously resolve  {signature} with candidates {candidates}. Please add {super_signature}.'.format(signature=self.signature, candidates=self.candidates, super_signature=super_signature(self.candidates)))

class Dispatcher(object):
    def __init__(self, name=None):
        self.functions = dict()
        self._cache = dict()
        self.signatures = []
        self.name = name

    def add(self, signature, function):
        signature = tuple(signature)
        if signature in self.functions:
            raise ValueError('cannot redefine function for signature {signature}, previous function {previous}'.format(signature=signature, previous=self.functions[signature]))
        self._invalidate(signature)
        self.signatures.append(signature)
        self.functions[signature] = function

    def _invalidate(self, signature):
        self._cache = dict()

    def _find(self, a):
        matches = []
        for signature in self.signatures:
            if len(signature) != len(a):
                continue
            if _supersedes(a, signature):
                if not any(map(lambda match:_supersedes(match, signature), matches)):
                    matches = [ match for match in matches if not _supersedes(signature, match)]
                    matches.append(signature)
        return matches

    def _call_slowpath(self, signature, args, kwargs):
        candidates = self._find(signature)
        if len(candidates) == 1:
            func = self.functions[candidates[0]]
        else:
            func = RaiseDispatchError(signature, candidates)
        self._cache[signature] = func
        return func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        signature = tuple([a.__class__ for a in args])
        try:
            return self._cache[signature](*args, **kwargs)
        except KeyError:
            return self._call_slowpath(signature, args, kwargs)
