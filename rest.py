#!/usr/bin/env python
# encoding: utf8
# Micro Restful API Framework

import sys
import re
import time
import datetime
import json
import inspect
import functools
import collections
import logging
import traceback


IS_PY3 = (sys.version_info >= (3, 0, 0))
if IS_PY3:
    basestring = str
    long = int
    bytes_to_str = lambda x: x.decode('utf8')
else:
    bytes = str
    bytes_to_str = lambda x: x


def with_metaclass(meta, *bases):
    return meta("NewBase", bases, {})


class DictObject(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


class CaseInsensitiveDict(DictObject):
    def __getitem__(self, key):
        return dict.__getitem__(self, str(key).title())

    def __setitem__(self, key, value):
        dict.__setitem__(self, str(key).title(), value)

    def __delitem__(self, key):
        dict.__delitem__(self, str(key).title())

    def __contains__(self, key):
        return dict.__contains__(self, str(key).title())

    def get(self, key, default=None):
        return dict.get(self, str(key).title(), default)

    def update(self, E):
        for k in E.keys():
            self[str(k).title()] = E[k]

    def pop(self, key, default):
        return dict.pop(self, str(key).title(), default)


def _get_real_fn(fn):
    while True:
        real_fn = getattr(fn, '_fn', None)
        if not real_fn:
            break
        fn = real_fn
    return fn


def _get_arg_info(fn):
    fn = _get_real_fn(fn)
    spec = inspect.getargspec(fn)
    info = DictObject()

    info.defaults = list(spec.defaults or [])
    info.args = spec.args
    info.varargs = spec.varargs
    info.keywords = spec.keywords

    required_count = len(info.args) - len(info.defaults)
    info.required_args = set(info.args[1:required_count])

    if 'self' in info.required_args:
        info.remove('self')

    info.default_dict = collections.OrderedDict(
        [(info.args[i + required_count], d)
         for i, d in enumerate(info.defaults)]
    )

    return info


def _arg_check(arg_info, input_args):
    if arg_info.required_args.issubset(input_args) and \
            (input_args.issubset(arg_info.args) or arg_info.keywords):
        return True
    return False


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


def be_strong(fn):
    real_fn = _get_real_fn(fn)
    arg_info = _get_arg_info(real_fn)
    if not arg_info or not arg_info.defaults:
        return fn

    strong_args = collections.OrderedDict()
    real_defaults = []
    for name, value in arg_info.default_dict.items():
        if not isinstance(value, StrongArg):
            real_defaults.append(value)
            continue

        strong_args[name] = value
        if not value.required:
            real_defaults.append(value.default)

        if real_defaults and value.required:
            raise SyntaxError(
                'non-default argument follows default argument')

    if not strong_args:
        return fn

    real_fn.__defaults__ = real_fn.func_defaults = tuple(real_defaults)
    fn._strong_args = real_fn._strong_args = strong_args

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        kwargs.update(dict(zip(arg_info.args, args)))

        for name, value in list(kwargs.items()):
            strong_arg = fn._strong_args.get(name)
            if not strong_arg:
                continue
            try:
                kwargs[name] = strong_arg.validate(value)
            except Exception as ex:
                raise StrongTypeError('%s(%s) type err: %s' % (name, value, ex))

        return fn(**kwargs)

    inner._fn = fn
    return inner


def as_method(method):
    def decorator(fn):
        fn._method = method
        return fn
    return decorator


def as_class(fn, method):
    return type(fn.func_name, (object,), {method.upper(): staticmethod(fn)})


def _json_loads(s):
    try:
        if isinstance(s, bytes):
            s = bytes_to_str(s).strip()
        return json.loads(s) if s else {}
    except Exception as ex:
        logging.error('json loads error: %s', ex)
        return {}


def _json_dumps_default(obj):
    if isinstance(obj, datetime.datetime):
        return int(time.mktime(obj.timetuple()))
    elif isinstance(obj, datetime.date):
        return obj.strftime('%Y-%m-%d')
    elif hasattr(obj, '_data'):
        return dict(obj._data)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


class RestError(Exception):
    type = 'BASE_ERROR'
    message = 'Base Rest Error Message'

    def __init__(self, detail=None):
        self.detail = detail

    def to_dict(self):
        return dict(type=self.type, message=self.message, detail=self.detail)


class DefaultError(RestError):
    type = 'DEFAULT_ERROR'
    message = 'API Error'


class StrongTypeError(RestError):
    type = 'STRONG_TYPE_ERROR'
    message = 'Strong Type Error'


class RestResourceMeta(type):

    def __new__(cls, name, bases, attrs):
        poly_methods = {}
        for k, v in attrs.items():
            if inspect.isfunction(v):
                v = be_strong(v)
                v._arg_info = _get_arg_info(v) # be_strong
                attrs[k] = v

                if hasattr(v, '_method'):
                    fs = poly_methods.get(v._method, [])
                    fs.append(v)
                    poly_methods[v._method] = fs

        for method, fs in poly_methods.items():
            if len(fs) == 1:
                attrs[method] = fs[0]
                continue

            fs.sort(key=lambda x: len(x._arg_info.required_args), reverse=True)

            # fuck Python closure using loop variant 
            def poly(self, _fs=fs, **kwargs):
                input_args = set(kwargs.keys())
                for f in _fs:
                    if _arg_check(f._arg_info, input_args):
                        return f(self, **kwargs)
                return None  # None means API not found

            poly._fs = fs
            attrs[method] = poly

        return type.__new__(cls, name, bases, attrs)


class RestResource(with_metaclass(RestResourceMeta)):
    def __init__(self, context=None):
        self.context = context
        self.extra_result = {}


class RestApp(object):

    def __init__(self,
                 mappings=None,
                 default_context=None,
                 ignore_more_args=True,
                 debug=False,
                 id_pattern='^\d+$',
                 path_prefix=None,
                 params_error=DefaultError,
                 noapi_error=DefaultError,
                 default_error=DefaultError,
                 ex_callback=None):

        self._mappings = dict(mappings or {})
        self._default_context = default_context or {}
        self._ignore_more_args = ignore_more_args
        self._debug = debug

        self._id_regexp = re.compile(id_pattern)
        self._path_prefix = path_prefix and path_prefix.strip('/').lower()

        self._params_error = params_error
        self._noapi_error = noapi_error
        self._default_error = default_error
        self.ex_callback = ex_callback

        self._debug = debug

    def parse_wsgi_environ(self, environ):

        def _parse_qs(s):
            kvs = [kv for kv in s.split('&') if kv.count('=') == 1]
            return dict([kv.split('=') for kv in kvs])

        method = environ['REQUEST_METHOD'].upper()
        path = environ['PATH_INFO'].lower()
        body = environ['wsgi.input'].read()
        query_str = environ['QUERY_STRING']
        headers = CaseInsensitiveDict()
        headers.update(
            {k[5:].replace('_', '-'): v
             for k, v in environ.items()
             if k.startswith('HTTP_')}
        )

        params = _json_loads(body)
        params.update(_parse_qs(query_str))
        return method, path, params, headers

    def make_context(self, params, headers):
        ctx = DictObject(self._default_context)
        ctx.update(dict(headers))
        return ctx

    def set_resp_headers(self, context):
        return []

    def wsgi(self, environ, start_response):
        method, path, params, headers = self.parse_wsgi_environ(environ)
        context = self.make_context(params, headers)
        resp = self.request(method, path, params, context)
        if self._debug:
            resp['context'] = dict(context)
            resp['headers'] = dict(headers)
        resp_headers = [('Content-Type', 'application/json')]
        resp_headers += self.set_resp_headers(context)
        status_line = '200 OK' if not resp['error'] else '400 Bad Request'
        start_response(status_line, resp_headers)
        return [json.dumps(resp, default=_json_dumps_default).encode('utf8')]

    def __call__(self, environ, start_response):
        return self.wsgi(environ, start_response)

    def run(self, host='0.0.0.0', port=8080, debug=None):
        self._debug = debug if debug is not None else self._debug
        from tornado import wsgi, httpserver, ioloop
        httpserver.HTTPServer(wsgi.WSGIContainer(self.wsgi)).listen(port, host)
        ioloop_ins = ioloop.IOLoop.instance()
        if self._debug:
            from tornado import autoreload, log
            log.enable_pretty_logging()
            autoreload.start(ioloop_ins)
        ioloop_ins.start()

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
        if self._id_regexp.match(s):
            return s
        else:
            return None

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
        parts = list(filter(lambda x: len(x) > 0, parts))
        if not parts:
            return None, None, {}

        if self._path_prefix and parts[0].lower() != self._path_prefix:
            return None, None, {}

        if parts[-1].startswith('_'):
            method_override = parts.pop()[1:]

        if len(parts) % 2 == 0:
            extra_params['id'] = self.to_id(parts.pop())

        endpoint = parts.pop()
        extra_params.update(
            {parts[i * 2] + '_id': self.to_id(parts[i * 2 + 1])
             for i in range(len(parts) // 2)}
        )

        if None in list(extra_params.values()):
            return None, None, {}

        return endpoint, method_override, extra_params

    def request(self, method, path, params=None, context=None):
        params = params or {}
        endpoint, method_override, extra_params = self.extract_path(path)
        rclass = self._mappings.get(endpoint)
        if not rclass:
            return self.make_response(
                error=self._noapi_error('No Such API Endpoint'),
            )

        rc = rclass(context) if issubclass(rclass, RestResource) else rclass()
        method = method_override or method
        params.update(extra_params)

        to_call = getattr(rc, method.upper(), None)
        if not callable(to_call):
            return self.make_response(
                error=self._noapi_error('No Such API Call')
            )

        arg_info = getattr(to_call, '_arg_info', _get_arg_info(to_call))
        input_args = set(params.keys())
        if not _arg_check(arg_info, input_args):
            more_args = input_args - set(arg_info.args)
            if self._ignore_more_args:
                for arg in more_args:
                    params.pop(arg)
            else:
                return self.make_response(error=self._params_error(
                    'ErrorArgs: %s' % ','.join(more_args)
                ))

        result, error = None, None
        try:
            result = to_call(**params)
        except StrongTypeError as ex:
            error = self._params_error(ex.detail)
        except RestError as ex:
            error = ex
        except Exception as ex:
            if isinstance(ex, TypeError) and \
                'required positional argument' in str(ex):
                error = self._params_error(str(ex))
            else:
                ex_detail = traceback.format_exc()
                ex_detail_list = ex_detail.split('\n')
                logging.error(ex_detail_list)
                error_detail = dict(ex=repr(ex), detail=ex_detail) if self._debug else repr(ex)
                error = self._default_error(error_detail)
                if self.ex_callback and callable(self.ex_callback):
                    self.ex_callback(ex_detail)

        if result is None and error is None:
            error = self._noapi_error('No Such API Result')

        extra_result = getattr(rc, 'extra_result', None)
        response = self.make_response(result, extra_result, error)
        return response


class RestClient(object):

    def __init__(self, base_url, headers=None, inject_params=None):
        import requests
        self._base_url = base_url.rstrip('/')
        self._headers = headers or {}
        self._inject_params = inject_params or {}
        self._session = requests.Session()

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
