import codecs
p = r'd:\Coding Section\Mediscan\templates\home.html'
with codecs.open(p,'w','utf-8') as f:
    f.write('TEST_WRITTEN_OK')
print('done')
