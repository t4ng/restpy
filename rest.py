#!/usr/bin/env python
# encoding: utf-8
# Micro Restful API Framework

import json
import inspect
import functools
import collections

class PolyMethod(object):
    def __init__(self, name):
        self._name = name
        self._fs = []

    def __call__(self, *args, **kwargs):
        if len(self._fs) == 1:
            return self._fs[0](*args, **kwargs)
        cargs = set(kwargs.keys())
        for f in self._fs:
            if f._required_args.issubset(cargs) and (f._keywords or cargs.issubset(f._args)):
                return f(*args, **kwargs)
        raise TypeError('%s got unexcept keyword argument %s'%(self._name, ','.join(args)))

    def add(self, f):
        self._fs.append(f)
        self._fs.sort(key=lambda x: len(x._required_args), reverse=True)

def as_method(method):
    def decorator(f):
        spec = inspect.getargspec(f)
        defaults = list(spec.defaults) if spec.defaults else []
        defaults = filter(lambda x: not getattr(x, 'required', False), defaults)

        f._method = method
        f._required_args = set(spec.args[1:len(spec.args)-len(defaults)])
        f._args = spec.args
        f._keywords = spec.keywords
        return f

    return decorator

def as_class(f, method):
    return type(f.func_name, (object,), {method.upper(): staticmethod(f)})

class StrongArg(object):
    type = object

    class NOT_SET(object):
        pass

    def __init__(self, default=NOT_SET, desc='-'):
        self.default = default
        self.required = (default == self.NOT_SET)
        self.desc = desc

    def validate(self, value):
        if isinstance(value, self.type):
            return value
        return self.type(value)

class IntArg(StrongArg):
    type = int

class StrArg(StrongArg):
    type = str

class BoolArg(StrongArg):
    type = bool

class ListArg(StrongArg):
    type = list

    def validate(self, value):
        if not isinstance(value, list):
            raise ValueError('Except list')
        return value

class IntListArg(ListArg):
    def validate(self, value):
        value = value if isinstance(value, list) else list(value)
        return map(lambda x: x if isinstance(x, int) else int(x), value)

class StrListArg(ListArg):
    def validate(self, value):
        value = value if isinstance(value, list) else list(value)
        return map(lambda x: x if isinstance(x, basestring) else str(x), value)

def be_strong(f):
    try: spec = inspect.getargspec(f)
    except: spec = None
    if not spec or not spec.defaults:
        return f

    strong_args = collections.OrderedDict()
    real_defaults = []
    required_count = len(spec.args) - len(spec.defaults)
    for i, d in enumerate(spec.defaults):
        if isinstance(d, StrongArg):
            arg_name = spec.args[required_count + i]
            strong_args[arg_name] = d
            if not d.required:
                real_defaults.append(d.default)
            if real_defaults and d.required:
                raise SyntaxError('non-default argument follows default argument')
        else:
            real_defaults.append(d)

    if not strong_args:
        return f

    real_defaults = tuple(real_defaults)
    func = getattr(f, '__func__', f)
    func.func_defaults = func.__defaults__ = real_defaults

    f._strong_args = strong_args
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs.update(dict(zip(spec.args, args)))
        for name, value in kwargs.items():
            strong_arg = f._strong_args.get(name)
            if strong_arg:
                kwargs[name] = strong_arg.validate(value)
        return f(**kwargs)

    for p in dir(f):
        if p[0] == '_' and p[1] != '_':
            setattr(wrapper, p, getattr(f, p))
    return wrapper

class RestError(Exception):
    type = 'BASE_ERROR'
    message = 'Base Rest Error Message'

    def __init__(self, detail=None):
        self.detail = detail

    def to_dict(self):
        return dict(type=self.type, message=self.message, detail=self.detail)

class RestResourceMeta(type):
    def __new__(cls, name, bases, attrs):
        for k, v in attrs.items():
            if inspect.isfunction(v):
                attrs[k] = be_strong(v)
        return type.__new__(cls, name, bases, attrs)

class RestResource(object):
    __metaclass__ = RestResourceMeta

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
            validator = getattr(self, '_validate_'+name, None)
            if validator: params[name] = validator(params[name])

def _parse_wsgi_environ(environ):
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
        req = _parse_wsgi_environ(environ)
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
        if not parts:
            return None, None, {}

        if parts[-1].startswith('_'):
            method_override = parts.pop()[1:]
        if len(parts) % 2 == 0:
            extra_params['id'] = self.to_id(parts.pop())
        endpoint = parts.pop()
        extra_params.update(
            {parts[i*2]+'_id': self.to_id(parts[i*2+1]) for i in range(len(parts)/2)}
        )
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
        result, error = None, None
        try:
            if isinstance(resource, RestResource):
                resource.validate_params(params)
            result = to_call(**params)
        except RestError as ex:
            error = ex
        extra_result = getattr(resource, 'extra_result', None)
        response = self.make_response(result, extra_result, error)
        return response

    def get_schema(self):
        apis = []
        for endpoint, rclass in self.mapping.iteritems():
            fs = [getattr(rclass, p) for p in dir(rclass) if not p.startswith('_')]
            fs = filter(lambda x: hasattr(x, '__func__') and \
                hasattr(x, '_strong_args') and \
                (x.__func__.__name__.isupper() or hasattr(x, '_method')), fs)
            for f in fs:
                method = getattr(f, '_method', f.__func__.__name__)
                parameters = []
                for arg_name, strong_arg in f._strong_args.iteritems():
                    parameters.append({
                        'field': arg_name,
                        'type': str(strong_arg.type).strip('<>'),
                        'optional': not strong_arg.required,
                        'description': strong_arg.desc,
                    })
                path = [k[:-3]+'/:'+k for k, v in f._strong_args.items() if k.endswith('_id') and v.required]
                path.append(endpoint)
                if 'id' in f._strong_args:
                    path.append(':id')
                if method not in {'GET', 'POST', 'PUT', 'PATCH', 'DELETE'}:
                    path.append('_' + method.lower())
                    method = 'POST'
                api = {
                    'group': endpoint,
                    'groupTitle': endpoint.upper(),
                    'type': method,
                    'name': f.__func__.__name__,
                    'title': '%s %s' % (method, endpoint),
                    'url': '/' + '/'.join(path) + '/',
                    'description': f.__doc__ or '',
                    'parameter': {'fields': {'Parameter': parameters}},
                }
                apis.append(api)
        return {'api': apis}

class RestClient(object):
    def __init__(self, base_url, headers={}):
        import requests
        self.base_url = base_url.rstrip('/')
        self.headers = headers
        self.session = requests.Session() # no retry

    def request(self, method, path, params=None):
        url = self.base_url + '/' + path
        params = params or {}
        if method != 'GET':
            params, data = None, json.dumps(params)
        else:
            params, data = params, None

        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=self.headers
            ).json()
        except:
            resp = {}
        return resp

    def __getattr__(self, name):
        return lambda x, y=None: self.request(name, x, y)

def _transform(s, vars):
    builtin_vars = {
        'true': True,
        'false': False,
        'none': None,
    }
    if s.lower() in builtin_vars:
        return builtin_vars[s.lower()]
    elif s.startswith('$'):
        ss = s.split('.')
        v = vars[ss[0][1:]]
        for k in ss[1:]:
            if k == 'length':
                return len(v)
            k = int(k) if k[1:].isdigit() else k
            v = v.__getitem__(k)
        return v
    elif s.isdigit():
        return int(s)
    else:
        return s

def TEST(expr):
    caller_frame = inspect.stack()[1][0]
    vars = caller_frame.f_locals
    A, op, B = map(lambda x: _transform(x, vars), expr.split(' '))
    ok = eval('A %s B' % op)
    tips = '\033[32m[OK]\033[0m' if ok else '\033[31m[FAIL]\033[0m'
    print('- %s -> %s'%(expr, tips))
    return ok, A, op, B


