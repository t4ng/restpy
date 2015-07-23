# Restpy
Micro Restful API Framework in Python. Simple, Easy and Extensible

## Quickstart
<pre>
class UserResource(rest.RestResource):
    def GET(self, id):
        return {'id': id, 'username': 'bitch'}
    
app = rest.RestApp()
app.map('user', UserResource)
app.run(port=80)
</pre>

## Multi-Level Endpoint
For example: when request **GET /user/123/blog/**, you need **user_id** in arguments
<pre>
class BlogResource(rest.RestResource):
    def GET(self, user_id):
        return [{'id':1}, {'id':2}]
</pre>

## Method Dispatch
Usually, we should support multi request like:
- GET /blog/?title=hello
- GET /blog/123/ or GET /blog/?id=123
- GET /user/123/blog/

you need **as_method** to dispatch these request to different methods:

<pre>
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
</pre>

## Custom Method
In Restful, there's only standard HTTP Method is supported, but sometimes it's hard to use GET/POST/PUT/PATCH/DELETE to describe our API, such as digg/follow. A simple solution is add them in then end of path, for example
- POST /blog/123/digg
- POST /user/123/follow

but its hard to distinguish them with resouce endpoint. So we make a convention: all the custom method shoud starts with underline, like this:
- POST /blog/123/_digg
- POST /user/123/_follow

then Restpy will route it to the correct method in resource class.

<pre>
class BlogResource(rest.RestResource):
    def DIGG(self, id):
        return {'is_digg': True}

class UserResource(rest.RestResource):
    def FOLLOW(self, id):
        return {'is_follow': True}
</pre>

## Embed Restpy in Django/Flask
The RestApp's entry is defined like this:

<pre>
class RestApp(object):
    def request(self, method, path, params, context=None):
        ...
</pre>

you can call it directly, it will return a dict response:

<pre>
app.request('GET', '/user/123/blog/', {'title':'haha'})
''' Response:
{
    'success': True,
    'error': None,
    'result': {'id': 111},
}'''
</pre>

So its very easy to embed it anywhere, you just need add a wildcard route in MVC framework, and parse method&path&params from MVC request. For example:

<pre>
# route: url(r'^api/(?P<path>.+)$', 'django_view'),
def django_view(request, path):
    method = request.method
    if method == 'GET':
        params = dict(request.GET.items())
    else:
        params = json.loads(request.raw_post_data)
    json_response = app.request(method, path, params)
    return json.dumps(json_response)
</pre>

## Use context
Sometimes, we need some context besides method/path/params, such as cookies/session and so on. In Restpy, you can deside what kind of context to use.
If you are using RestApp as a single WSGI Application, you need inherit RestApp and override **extract_wsgi_environ** method:

<pre>
# original extract_wsgi_environ, last return value is context
def extract_wsgi_environ(self, environ):
    req = parse_wsgi_environ(environ)
    return req['method'], req['path'], req['params'], req['headers']
    
# custom context 
def extract_wsgi_environ(self, environ):
    req = parse_wsgi_environ(environ)
    return req['method'], req['path'], req['params'], {'hello':'bitch'}
</pre>

If you are using RestApp in other MVC Framework, just pass context when you call **app.request**

<pre>
app.request(method, path, params, context={'hello':'bitch'})
</pre>

Use context in resource class method:

<pre>
class BlogResource(rest.RestResource):
    def GET(self, id):
        ctx = self.context
        return {'id': id, 'current_user_id': ctx.user_id}
</pre>
    
## Other features
### API Errors
<pre>
class ParamsError(rest.RestError):
    type = 'PARAMS_ERROR'
    message = 'Params Error'

class UserResource(rest.RestResource):
    def GET(self, id):
        if not id:
            raise ParamsError()
        return {'id':id}
</pre>

### Decorator Map
<pre>
@app.mapper('user')
class UserResource(rest.RestResource):
    ...
</pre>

### Function Map
<pre>
@app.mapper('user', method='GET')
def get_user(id):
    return {'id': id}
</pre>

### Extra Result
<pre>
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
</pre>

### Custom Response Format
<pre>
class MyRestApp(rest.RestApp):
    def make_response(self, result=None, extra_result=None, error=None):
        return {
            'status': 'success' if not error else 'error',
            'error': error.to_dict(),
            'result': result,
            'extra_result': extra_result,
        }
</pre>

### Custom Id Format
Default Id format is number, if you want use hex format:
<pre>
class MyRestApp(rest.RestApp):
    def to_id(self, s):
        if len(s) != 32 or not is_hex(s):
            return None
        return s
</pre>






