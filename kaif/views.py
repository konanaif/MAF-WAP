from django.http import HttpResponse

import os

# Create your views here.

def index(request):
	page = """<h1>Main page</h1>
<p>Choose data type</p>
<ul>
	<li><a href="audio/">Audio</a></li>
	<li><a href="image/">Image</a></li>
	<li><a href="raw/">Raw</a></li>
	<li><a href="text/">Text</a></li>
	<br><br>
	<li><a href="new/">New algorithms</a></li>
</ul>"""
	return HttpResponse(page)