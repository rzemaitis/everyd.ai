import configparser


def read_config(config_file):
    '''
    Reads in a configuration file.

    Parameters:
    - config_file: Name of the configuration file.

    Returns:
    A ConfigParser instance with read in configuration.
    '''
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def create_config(loc):
    '''
    Creates a configuration file 'config.txt' with predefined settings
    and writes it to the specified location.

    Parameters:
    - loc: A string representing the directory path where the
    configuration file will be saved.

    Returns:
    None
    '''
    config = configparser.ConfigParser()
    config['main'] = {'extension': '.jpg',
                      'datestart': '2009-07-15',
                      'datefinish': '2023-09-21'
                      }
    config['directories'] = {'imgdir': './Images',
                             'moviedir': './Movies',
                             'reviewdir': './Review',
                             'purgedir': './Purged',
                             'solutionsdir': './Solutions'
                             }
    config['dimensions'] = {'template': './template_image.jpg',
                            'dimfile': '',
                            'height': '',
                            'width': ''
                            }
    config['multiprocessing'] = {'cores': '1'}

    config['review'] = {'sleepstart': '3',
                        'sleepfinish': '7',
                        'puttemplate': 'True'
                        }

    with open(loc+'config.txt', 'w') as configfile:
        config.write(configfile)
