#!/usr/bin/env python
# encoding: utf-8
# Micro Restful API Framework

import time
import datetime
import json
import base64
import inspect
import functools
import collections


class StrongArg(object):
    type = object

    class NOT_SET(object):
        pass

    def __init__(self, default=NOT_SET, desc='-'):
        self.default = default
        self.required = (default == self.NOT_SET)
        self.desc = desc

    def validate(self, value):
        return value if isinstance(value, self.type) else self.type(value)


class IntArg(StrongArg):
    type = int

    def validate(self, value):
        return value if isinstance(value, (int, long)) else int(value)


class FloatArg(StrongArg):
    type = float


class StrArg(StrongArg):
    type = str


class BoolArg(StrongArg):
    type = bool


class ListArg(StrongArg):
    type = list

    def validate(self, value):
        if not isinstance(value, list):
            return list(value)
        return value


class IntListArg(ListArg):

    def validate(self, value):
        value = value if isinstance(value, list) else list(value)
        return map(lambda x: x if isinstance(x, (int, long)) else int(x), value)


class StrListArg(ListArg):

    def validate(self, value):
        value = value if isinstance(value, list) else list(value)
        return map(lambda x: x if isinstance(x, basestring) else str(x), value)


def be_strong(f):
    spec = inspect.getargspec(f)
    if not spec or not spec.defaults:
        return f

    required_count = len(spec.args) - len(spec.defaults)
    strong_args = collections.OrderedDict()
    real_defaults = []

    for i, d in enumerate(spec.defaults):
        if not isinstance(d, StrongArg):
            real_defaults.append(d)
            continue

        arg_name = spec.args[required_count + i]
        strong_args[arg_name] = d

        if not d.required:
            real_defaults.append(d.default)

        if real_defaults and d.required:
            raise SyntaxError(
                'non-default argument follows default argument')

    if not strong_args:
        return f

    real_defaults = tuple(real_defaults)
    func = getattr(f, '__func__', f)
    func.func_defaults = func.__defaults__ = real_defaults

    f._strong_args = strong_args

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs.update(dict(zip(spec.args, args)))

        for name, value in list(kwargs.items()):
            strong_arg = f._strong_args.get(name)
            if strong_arg:
                kwargs[name] = strong_arg.validate(value)  # TODO Try:TypeError

        return f(**kwargs)

    wrapper._func = f
    return wrapper


def as_method(method):
    def decorator(f):
        f._method = method
        return f
    return decorator


def as_class(f, method):
    return type(f.func_name, (object,), {method.upper(): staticmethod(f)})


class RestError(Exception):
    type = 'BASE_ERROR'
    message = 'Base Rest Error Message'

    def __init__(self, detail=None):
        self.detail = detail

    def to_dict(self):
        return dict(type=self.type, message=self.message, detail=self.detail)


class RestResourceMeta(type):

    def __new__(cls, name, bases, attrs):
        poly_methods = {}
        for k, v in attrs.items():
            if inspect.isfunction(v):
                v = be_strong(v)
                attrs[k] = v
                if hasattr(v, '_method'):
                    fs = poly_methods.get(v._method, [])
                    fs.append(v)
                    poly_methods[v._method] = fs

        for method, fs in poly_methods.items():
            if len(fs) == 1:
                attrs[method] = fs[0]
                continue

            for f in fs:
                spec = inspect.getargspec(getattr(f, '_func', f)) # f is wrapper in be_strong
                defaults = list(spec.defaults) if spec.defaults else []

                f._required_args = set(
                    spec.args[1:len(spec.args) - len(defaults)])
                f._args = spec.args
                f._keywords = spec.keywords

            fs.sort(key=lambda x: len(x._required_args), reverse=True)

            def poly(self, *args, **kwargs):
                cargs = set(kwargs.keys())
                for f in fs:
                    if f._required_args.issubset(cargs) and \
                            (cargs.issubset(f._args) or f._keywords):
                        return f(self, *args, **kwargs)
                raise TypeError('%s got unexcept keyword argument %s' %
                                (self._name, ','.join(args)))

            poly._fs = fs
            attrs[method] = poly

        attrs['_poly_methods'] = poly_methods
        return type.__new__(cls, name, bases, attrs)


def _json_loads(s):
    try:
        return json.loads(s)
    except:
        return {}


def _json_dumps_default(obj):
    if isinstance(obj, datetime.datetime):
        return int(time.mktime(obj.timetuple()))
    elif isinstance(obj, datetime.date):
        return obj.strftime('%Y-%m-%d')
    else:
        return obj.__dict__


class RestResource(object):
    __metaclass__ = RestResourceMeta

    def __init__(self, context=None):
        self.context = context
        self.extra_result = {}


class RestContext(dict):

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class RestApp(object):

    def __init__(self, mappings=None, default_context=None):
        self._mappings = dict(mappings) if mappings else {}
        self._default_context = default_context or {}

    def parse_wsgi_environ(self, environ):

        def _parse_qs(s):
            kvs = [kv for kv in s.split('&') if kv.count('=') == 1]
            return dict([kv.split('=') for kv in kvs])

        method = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'].lower()
        body = environ['wsgi.input'].read()
        query_str = environ['QUERY_STRING']
        headers = {k[5:]: v
                   for k, v in environ.items() 
                   if k.startswith('HTTP_')}

        params = _json_loads(body)
        params.update(dict([kv.split('=')
                            for kv in query_str.split('&')
                            if kv.count('=') == 1]))
        return method, path, params, headers

    def headers_to_context(self, headers):
        ctx_header = headers.get('X-Rest-Context', '')
        ctx_header = base64.b64decode(ctx_header)
        ctx = RestContext(self._default_context)
        ctx.update(_json_loads(ctx_header))
        return ctx

    def wsgi(self, environ, start_response):
        method, path, params, headers = self.parse_wsgi_environ(environ)
        context = self.headers_to_context(headers)
        resp = self.request(method, path, params, context)
        start_response('200 OK', [('Content-Type', 'application/json')])
        return [json.dumps(resp, default=_json_dumps_default, indent=2)]

    def run(self, host='0.0.0.0', port=8080):
        from tornado import wsgi, httpserver, ioloop
        httpserver.HTTPServer(wsgi.WSGIContainer(self.wsgi)).listen(port, host)
        ioloop.IOLoop.instance().start()

    def map(self, endpoint, rclass):
        self._mappings[endpoint] = rclass

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

    def make_response(self, result=None, extra_result=None, error=None):
        return {
            'success': error is None,
            'error': error.to_dict() if error else None,
            'result': result,
            'extra_result': extra_result,
        }

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
            {parts[i * 2] + '_id': self.to_id(parts[i * 2 + 1])
             for i in range(len(parts) / 2)}
        )

        if None in extra_params.values():
            return None, None, {}

        return endpoint, method_override, extra_params

    def request(self, method, path, params=None, context=None):
        params = params or {}
        endpoint, method_override, extra_params = self.extract_path(path)
        rclass = self._mappings.get(endpoint)
        if not rclass:
            return {}

        resource = rclass(context) if issubclass(
            rclass, RestResource) else rclass()
        method = method_override or method
        params.update(extra_params)

        to_call = getattr(resource, method.upper(), None)
        if not callable(to_call):
            return {}

        result, error = None, None
        try:
            result = to_call(**params)  # TODO
        except RestError as ex:
            error = ex

        extra_result = getattr(resource, 'extra_result', None)
        response = self.make_response(result, extra_result, error)
        return response


class RestClient(object):

    def __init__(self, base_url, headers=None, inject_params=None):
        import requests
        self._base_url = base_url.rstrip('/')
        self._headers = headers or {}
        self._inject_params = inject_params or {}
        self._session = requests.Session()  # no retry

    def request(self, method, path, params=None):
        url = self._base_url + '/' + path.lstrip('/')
        params = params or {}

        if method != 'GET':
            params, data = {}, json.dumps(params)
        else:
            params, data = params, None

        params.update(self._inject_params)

        try:
            resp = self._session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                headers=self._headers
            ).json()
        except:
            resp = {}

        return resp

    def __getattr__(self, name):
        return lambda x, y=None: self.request(name, x, y)


def TEST(expr):
    ok = expr
    tips = '\033[32m[OK]\033[0m' if ok else '\033[31m[FAIL]\033[0m'
    frame, _, _, _, lines, _ = inspect.stack(2)[1]
    pre_line = lines[0].strip()
    if not pre_line.startswith('TEST') and pre_line.find('/') > 0:
        print('=' * 80)
        print(pre_line[pre_line.find('=') + 1:].strip())
    expr = lines[1].strip()
    expr = expr[expr.find('('):].strip('()')
    print('- %s -> %s' % (expr, tips))
    if not ok:
        locals().update(frame.f_locals)
        import pdb
        pdb.set_trace()
