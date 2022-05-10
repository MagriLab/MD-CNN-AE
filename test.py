import configparser

from git import safe_decode
config = configparser.ConfigParser()
config.read('__system.ini')
print(config.sections())
system_info = config['system_info']
print(type(system_info['save_location']))
print(system_info['save_location'])
print('/hi')
print(system_info['save_location']+'/hi')