from django.conf.urls import patterns, include, url
from django.contrib import admin
import settings




# stuff



urlpatterns = patterns('',
    # Examples:
     url(r'^$', 'app1.views.query', name='query'),
     url(r'^goto/','app1.views.query_1', name='query_1' ),
	 #url(r'^image/','app1.views.image', name='image' ),
	 #url(r'^static/(?P<path>.*)$', 'django.views.static.serve',{'document_root': settings.MEDIA_ROOT}),
	 
	 

    url(r'^admin/', include(admin.site.urls)),
)
