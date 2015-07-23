#!/usr/bin/env python
# encoding: utf-8
# Micro Restful API Framework

import json
import inspect

class RestError(Exception):
    type = 'BASE_ERROR'
    message = 'Base Rest Error Message'

    def __init__(self, detail=None):
        self.detail = detail

    def to_dict(self):
        return dict(type=self.type, message=self.message, detail=self.detail)

class PolyMethod(object):
    def __init__(self, name):
        self._name = name
        self._fs = []

    def __call__(self, *args, **kwargs):
        if len(self._fs) == 1:
            return self._fs[0](*args, **kwargs)
        for f in self._fs:
            if f._required_args.issubset(set(kwargs.keys())):
                return f(*args, **kwargs)
        raise TypeError('%s got unexcept keyword argument %s'%(self._name, ','.join(args)))

    def add(self, f):
        self._fs.append(f)
        self._fs.sort(key=lambda x:len(x._required_args), reverse=True)

class RestResource(object):
    def __init__(self, context=None):
        self.context = context
        self.extra_result = {}

        fs = [getattr(self, p) for p in dir(self)]
        fs = [f for f in fs if hasattr(f, '_method')]
        for f in fs:
            ploy_method = getattr(self, f._method, PolyMethod(f._method))
            ploy_method.add(f)
            setattr(self, f._method, ploy_method)

    def validate_params(self, params):
        for name in params.keys():
            validater = getattr(self, '_validate_'+name, None)
            if validater: params[name] = validater(params[name])

def as_method(method):
    def decorator(f):
        spec = inspect.getargspec(f)
        end_idx = -len(spec.defaults) if spec.defaults else len(spec.args)
        f._method = method
        f._required_args = set(spec.args[1:end_idx])
        return f
    return decorator

def as_class(f, method):
    return type(f.func_name, (object,), {method.upper():staticmethod(f)})

def parse_wsgi_environ(environ):
    def _parse_json(s):
        try: return json.loads(s)
        except: return {}

    def _parse_qs(s):
        kvs = [kv for kv in s.split('&') if kv.count('=')==1]
        return dict([kv.split('=') for kv in kvs])

    method = environ['REQUEST_METHOD'].upper()
    path = environ['PATH_INFO'].lower()
    body = environ['wsgi.input'].read()
    query_str = environ['QUERY_STRING']
    headers = {k[5:]: v for k, v in environ.iteritems() if k.startswith('HTTP_')}

    params = _parse_json(body)
    params.update(_parse_qs(query_str))
    return dict(method=method, path=path, params=params, headers=headers)

class RestApp(object):
    def __init__(self, mapping={}):
        self.mapping = dict(mapping)

    def extract_wsgi_environ(self, environ):
        req = parse_wsgi_environ(environ)
        return req['method'], req['path'], req['params'], req['headers']

    def wsgi(self, environ, start_response):
        method, path, params, context = self.extract_wsgi_environ(environ)
        resp = self.request(method, path, params, context)
        start_response('200 OK', [('Content-Type', 'application/json')])
        return [json.dumps(resp, indent=2)]

    def run(self, host='0.0.0.0', port=8080):
        from tornado import wsgi, httpserver, ioloop
        httpserver.HTTPServer(wsgi.WSGIContainer(self.wsgi)).listen(port, host)
        ioloop.IOLoop.instance().start()

    def map(self, endpoint, rclass):
        self.mapping[endpoint] = rclass

    def mapper(self, endpoint, method=None):
        def decorator(func_or_class):
            if method and callable(func_or_class):
                self.map(endpoint, as_class(func_or_class, method))
            else:
                self.map(endpoint, func_or_class)
            return func_or_class
        return decorator

    def to_id(self, s):
        return int(s) if s.isdigit() else None

    def extract_path(self, path):
        endpoint, method_override, extra_params = None, None, {}
        parts = [part.strip().lower() for part in path.split('/')]
        parts = filter(lambda x: len(x) > 0, parts)
        if not parts: return None, None, {}

        if parts[-1].startswith('_'):
            method_override = parts.pop()[1:]
        if len(parts) % 2 == 0:
            extra_params['id'] = self.to_id(parts.pop())
        endpoint = parts.pop()
        extra_params.update({parts[i*2]+'_id': self.to_id(parts[i*2+1]) for i in range(len(parts)/2)})
        if None in extra_params.values():
            return None, None, {}

        return endpoint, method_override, extra_params

    def make_response(self, result=None, extra_result=None, error=None):
        return {
            'success': error is None,
            'error': error.to_dict() if error else None,
            'result': result,
            'extra_result': extra_result,
        }

    def request(self, method, path, params=None, context=None):
        params = params or {}
        endpoint, method_override, extra_params = self.extract_path(path)
        rclass = self.mapping.get(endpoint)
        if not rclass:
            return {}

        resource = rclass(context) if issubclass(rclass, RestResource) else rclass()
        method = method_override or method
        params.update(extra_params)
        to_call = getattr(resource, method.upper(), None)
        if not callable(to_call):
            return {}

        if isinstance(resource, RestResource):
            resource.validate_params(params)
        result, error = None, None
        try:
            result = to_call(**params)
        except RestError as ex:
            error = ex
        extra_result = getattr(resource, 'extra_result', None)
        response = self.make_response(result, extra_result, error)
        return response
