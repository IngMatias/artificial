from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('<str:section_path>/<str:entry_path>', views.get_template),
    path('linkedin', views.linkedin, name='linkedin')
]