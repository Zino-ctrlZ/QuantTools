from trade.assets.Option import Option
from trade.assets.OptionStructure import OptionStructure
from datetime import datetime
from trade.helpers.helper import parse_option_tick
from trade.helpers.Context import Context

def create_object_from_id(option_id: str, 
                          date: str = datetime.today().strftime('%Y-%m-%d'), 
                          run_chain: bool = False):
    
    """
    return an Option object from an option id, or an OptionStructure object from a structure id

    Args:
    option_id: str: the option id to create an object from
    date: str: the date to use for the option object build date
    run_chain: bool: whether to run the chain for the option object

    Returns:
    Option or OptionStructure: the object created from the option id
    
    """
    if 'L:' in option_id or 'S:' in option_id:
        structure = {'long': [], 'short': []}
        split1 = option_id.split('&')
        split2 = [x.split(':') for x in split1]
        split2 = [x if x != [''] else None for x in split2]
        for option in split2:
            if option is not None:
                details = parse_option_tick(option[1])
                if option[0] == 'L':
                    side = 'long'
                elif option[0] == 'S':
                    side = 'short'
                else:
                    raise ValueError('Invalid option structure')
                
                structure[side].append({
                    'strike': details['strike'],
                    'expiration': details['exp_date'],
                    'right': details['put_call'].lower(),
                    'underlier': details['ticker']
                })

        with Context(end_date=date) as ctx:
            structure = OptionStructure(structure, run_chain = run_chain)
        return structure
    else:
        details = parse_option_tick(option_id)
        with Context(end_date=date) as ctx:
            return Option(**details, run_chain = run_chain)