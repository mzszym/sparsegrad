# -*- coding: utf-8; -*-
#
# sparsegrad - automatic calculation of sparse gradient
# Copyright (C) 2016-2018 Marek Zdzislaw Szymanski (marek@marekszymanski.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License, version 3,
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__all__ = ['GenericFunction', 'GenericMethod', 'Dispatcher',
           'DispatchError', 'dispatch', 'dispatchmethod']


def supersedes(a, b):
    """a is strictly more specific than b"""
    if len(a) != len(b):
        return False
    return all(map(issubclass, a, b))


def _supersedes(a, b):
    return all(map(issubclass, a, b))


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
        if len(self.candidates):
            raise DispatchError(
                'Cannot unambigously resolve {signature} with candidates {candidates}. Please add {super_signature}.'.format(
                    signature=self.signature,
                    candidates=self.candidates,
                    super_signature=super_signature(
                        self.candidates)))
        else:
            raise DispatchError('Function does not have a dispatch candidate. Please add {signature}'.format(
                signature=self.signature))


def dispatchSignature(values):
    return tuple([a.__class__ for a in values])


class Dispatcher(object):
    def __init__(self, name=None, doc=None):
        self.functions = dict()
        self._cache = dict()
        self.signatures = []
        self.name = name
        self.doc = doc

    def add(self, signature, function):
        signature = tuple(signature)
        if signature in self.functions:
            raise ValueError(
                'cannot redefine function for signature {signature}, previous function {previous}'.format(
                    signature=signature, previous=self.functions[signature]))
        self._invalidate(signature)
        self.signatures.append(signature)
        self.functions[signature] = function

    def register(self, *types):
        return lambda function: self.add(types, function)

    def _invalidate(self, signature):
        self._cache = dict()

    def _find(self, a):
        matches = []
        for signature in self.signatures:
            if len(signature) != len(a):
                continue
            if _supersedes(a, signature):
                if not any(map(lambda match: _supersedes(
                        match, signature), matches)):
                    matches = [
                        match for match in matches if not _supersedes(
                            signature, match)]
                    matches.append(signature)
        return matches

    def _all_same(self, candidates):
        func = self.functions[candidates[0]]
        return all(self.functions[c] == func for c in candidates[1:])

    def _dispatch_slowpath(self, signature):
        candidates = self._find(signature)
        if len(candidates) == 1 or len(candidates) > 1 and self._all_same(candidates):
            func = self.functions[candidates[0]]
        else:
            func = RaiseDispatchError(signature, candidates)
        self._cache[signature] = func
        return func

    def _call_slowpath(self, signature, args, kwargs):
        func = self._dispatch_slowpath(signature)
        return func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        signature = dispatchSignature(args)
        try:
            return self._cache[signature](*args, **kwargs)
        except KeyError:
            pass
        return self._call_slowpath(signature, args, kwargs)

    def dispatch(self, *signature):
        try:
            return self._cache[signature]
        except KeyError:
            pass
        return self._dispatch_slowpath(signature)

    def __getstate__(self):
        return dict(functions=self.functions,
                    signatures=self.signatures, name=self.name, doc=self.doc)

    def __setstate__(self, state):
        self.name = state['name']
        self.functions = state['functions']
        self.signatures = state['signatures']
        self.doc = state['doc']

    @property
    def __doc__(self):
        lines = ['Multiply dispatched method {name}:'.format(name=self.name)]
        if self.doc is not None:
            lines.append('\t'.join(str(self.doc).splitlines()))
            lines.append('')
        lines = [
            '{count} known implentations:'.format(
                count=len(
                    self.signatures))]
        for index, signature in enumerate(self.signatures):
            function = self.functions[signature]
            lines.append(
                '{index} {signature}: '.format(
                    index=index,
                    signature=signature))
            if function.__doc__ is not None:
                lines.append('\t'.join(str(function.__doc__)).splitlines())
        return '\n'.join(lines)


class BoundMethod(object):
    def __init__(self, dispatcher, instance, owner):
        self.instance = instance
        self.owner = owner
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        return self.dispatcher.dispatch(
            *dispatchSignature(args))(self.instance, *args, **kwargs)

    def add(self, *args, **kwargs):
        self.dispatcher.add(*args, **kwargs)


class MethodDispatcher(Dispatcher):
    def __get__(self, instance, cls):
        return BoundMethod(self, instance, cls)


def GenericFunction(self, *args, **kwargs):
    return Dispatcher(*args, **kwargs)


def GenericMethod(self, *args, **kwargs):
    return MethodDispatcher(*args, **kwargs)


def _dispatch(types, dispatcher_type, kwargs):
    def _(func):
        result = dispatcher_type(func.__name__, **kwargs)
        result.add(types, func)
        return result
    return _


def dispatch(*types, **kwargs):
    return _dispatch(types, Dispatcher, kwargs)


def dispatchmethod(*types, **kwargs):
    return _dispatch(types, MethodDispatcher, kwargs)
