#!/usr/bin/env python
# encoding: utf-8

import rest

class ParamsError(rest.RestError):
    type = 'PARAMS_ERROR'
    message = 'Params Error'

app = rest.RestApp()

class UserResource(rest.RestResource):
    def GET(self, id):
        return {'id': id, 'username': 'Steve'}

    def POST(self, username, password):
        return {'id': 123}

class BlogResource(rest.RestResource):
    @rest.as_method('GET')
    def get_one(self, id):
        if not id:
            raise ParamsError()
        return {'id': id, 'title': 'First Blog', 'text': 'blabla..'}

    @rest.as_method('GET')
    def get_user_blogs(self, user_id):
        return [
            {'id': 1, 'title': 'aaa', 'text': '...'},
            {'id': 2, 'title': 'bbb', 'text': '...'},
        ]

    def DIGG(self, id):
        return {'id':id, 'is_digg':True}

# Route Detail:
# GET /user/123/ -> UserResource().GET(id=123)
# GET /blog/123/ -> BlogResource().get_one(id=123)
# POST /blog/123/_digg/ -> BlogResource().DIGG(id=123)
# GET /user/123/blog/ -> BlogResource().get_user_blogs(user_id=123)

app.map('user', UserResource) # or user decorator: @app.mapper('user') on UserResource instead
app.map('blog', BlogResource)

# embeded in django
def django_view(request, path):
    import json
    method = request.method
    if method == 'GET': params = dict(request.GET.items())
    else: params = json.loads(request.raw_post_data)
    resp = app.request(method, path, params, context=request)
    return json.dumps(resp, indent=2)

if __name__ == '__main__':
    app.run(port=9888)
