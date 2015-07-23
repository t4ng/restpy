# Restpy
Micro Restful API Framework in Python. Simple, Easy and Extensible

## Quickstart
```python
class UserResource(rest.RestResource):
    def GET(self, id):
        return {'id': id, 'username': 'bitch'}

app = rest.RestApp()
app.map('user', UserResource)
app.run(port=80)
```

## Multi-Level Endpoint
For example: when request **GET /user/123/blog/**, you need **user_id** in arguments
```python
class BlogResource(rest.RestResource):
    def GET(self, user_id):
        return [{'id':1}, {'id':2}]
```

## Method Dispatch
Usually, we should support multi request like:
- GET /blog/?title=hello
- GET /blog/123/ or GET /blog/?id=123
- GET /user/123/blog/

you need **as_method** to dispatch these request to different methods:
```python
class BlogResource(rest.RestResource):
    @rest.as_method('GET')
    def get_one(self, id):
        return {'id': id}
    
    @rest.as_method('GET')
    def get_by_title(self, title):
        return {'id':1, 'title':title}
        
    @rest.as_method('GET')
    def get_by_user(self, user_id):
        return [{'id':1}, {'id':2}]
```

## Custom Method
In Restful, there's only standard HTTP Method is supported, but sometimes it's hard to use GET/POST/PUT/PATCH/DELETE to describe our API, such as digg/follow. A simple solution is add them in then end of path, for example
- POST /blog/123/digg
- POST /user/123/follow

but its hard to distinguish them with resouce endpoint. So we make a convention: all the custom method shoud starts with underline, like this:
- POST /blog/123/_digg
- POST /user/123/_follow

then Restpy will route it to the correct method in resource class.

```python
class BlogResource(rest.RestResource):
    def DIGG(self, id):
        return {'is_digg': True}

class UserResource(rest.RestResource):
    def FOLLOW(self, id):
        return {'is_follow': True}
```

## Embed Restpy in Django/Flask
The RestApp's entry is defined like this:

```python
class RestApp(object):
    def request(self, method, path, params, context=None):
        ...
```

you can call it directly, it will return a dict response:

```python
app.request('GET', '/user/123/blog/', {'title':'haha'})
# Response:
{
    'success': True,
    'error': None,
    'result': {'id': 111},
}
```

So its very easy to embed it anywhere, you just need add a wildcard route in MVC framework, and parse method&path&params from MVC request. For example:

```python
# route: url(r'^api/(?P<path>.+)$', 'django_view'),
def django_view(request, path):
    method = request.method
    if method == 'GET':
        params = dict(request.GET.items())
    else:
        params = json.loads(request.raw_post_data)
    json_response = app.request(method, path, params)
    return json.dumps(json_response)
```

## Use context
Sometimes, we need some context besides method/path/params, such as cookies/session and so on. In Restpy, you can deside what kind of context to use.
If you are using RestApp as a single WSGI Application, you need inherit RestApp and override **extract_wsgi_environ** method:

```python
# original extract_wsgi_environ, last return value is context
def extract_wsgi_environ(self, environ):
    req = parse_wsgi_environ(environ)
    return req['method'], req['path'], req['params'], req['headers']
    
# custom context 
def extract_wsgi_environ(self, environ):
    req = parse_wsgi_environ(environ)
    return req['method'], req['path'], req['params'], {'hello':'bitch'}
```

If you are using RestApp in other MVC Framework, just pass context when you call **app.request**

```python
app.request(method, path, params, context={'hello':'bitch'})
```

Use context in resource class method:

```python
class BlogResource(rest.RestResource):
    def GET(self, id):
        ctx = self.context
        return {'id': id, 'current_user_id': ctx.user_id}
```
    
## Other features
### API Errors
```python
class ParamsError(rest.RestError):
    type = 'PARAMS_ERROR'
    message = 'Params Error'

class UserResource(rest.RestResource):
    def GET(self, id):
        if not id:
            raise ParamsError()
        return {'id':id}
```

### Decorator Map
```python
@app.mapper('user')
class UserResource(rest.RestResource):
    ...
```

### Function Map
```python
@app.mapper('user', method='GET')
def get_user(id):
    return {'id': id}
```

### Extra Result
```python
class BlogResource(rest.RestResource):
    def GET(self):
        self.extra_result['total'] = 100
        self.extra_result['page'] = 5
        return [{'id':1}, {'id':2}]
        
# Response:
{
    'success': True,
    'error': None,
    'result': [{'id':1}, {'id':2}],
    'extra_result': {
        'total': 100,
        'page': 5,
    },
}
```

### Custom Response Format
```python
class MyRestApp(rest.RestApp):
    def make_response(self, result=None, extra_result=None, error=None):
        return {
            'status': 'success' if not error else 'error',
            'error': error.to_dict(),
            'result': result,
            'extra_result': extra_result,
        }
```

### Custom Id Format
Default Id format is number, if you want use hex format:
```python
class MyRestApp(rest.RestApp):
    def to_id(self, s):
        if len(s) != 32 or not is_hex(s):
            return None
        return s
```

### Params Validate
define **_validate_x** function for param x
```python
class BlogResource(rest.Resource):
    _validate_id = int
    def _validate_count(self, count):
        return int(count)
```
