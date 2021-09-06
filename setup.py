import cx_Freeze
import sys
import matplotlib

base = None

if sys.platform == 'win32':
    base = 'Win32GUI'

executables = [cx_Freeze.Executable('Software.py', base=base)]

cx_Freeze.setup(
    name = 'Software',
    options = {'build_exe':{'packages':['tkinter', 'matplotlib']}},
    version = '0.01',
    description = 'Charting Software',
    executables = executables)