
from django.shortcuts import render, redirect
from lxml import html
from datetime import datetime as dt
import os

entries_path = './blog/templates/blog_entries'

def get_entry(section_path, entry_file):
  entry_path = os.path.join(entries_path, section_path, entry_file)
  entry = {}
  
  html_str = open(entry_path, 'r').read()
  html_tree = html.fromstring(html_str)

  entry['position'] = int(entry_file.split('_')[0])
  entry['icon'] = '/static/' + html_tree.xpath('//img[contains(@class, "icon")]/@src')[0].split('\'')[1]  
  entry['title'] =  html_tree.xpath('//h1/text()')[0]
  entry['last_modified_epoch'] = os.path.getmtime(entry_path)
  datetime =  dt.fromtimestamp(entry['last_modified_epoch'])
  entry['last_modified'] = datetime.strftime("%d %B %Y")
  entry['href'] = os.path.join(section_path, entry_file)

  return entry

def get_entries_by_sections():
  sections = []

  sections_dir = os.listdir(entries_path)
  for section_dir in sections_dir:
    entries_dir = os.listdir(os.path.join(entries_path, section_dir))

    entries = []
    for entry_file in entries_dir:
      entries.append(get_entry(section_dir, entry_file))

    section = {}
    section['position'] = int(section_dir.split('_')[0])
    section['title'] = ' '.join(section_dir.split('_')[1:])
    section['entries'] = sorted(entries, key=lambda x: -1 * x['position'])
    sections.append(section)

  return sorted(sections, key=lambda  x: -1 * x['position'])
  
def home(request):
  sections = get_entries_by_sections()
  return render(request, 'home.html', {
    'title': 'Home | Artificial Intelligence | Matias Hernandez',
    'sections': sections
  })
  
def get_template(request, section_path, entry_path):
  entry = get_entry(section_path, entry_path)
  return render(request, os.path.join('blog_entries', section_path, entry_path), {
    'title': entry['title'],
    'last_modified': entry['last_modified']
  })

def linkedin(request):
  return redirect("https://www.linkedin.com/in/matias-hernandezf")
