import sys, os
from dotenv import load_dotenv
# sys.path.extend([os.environ['WORK_DIR'], os.environ['DBASE_DIR']])
from trade.assets.Option import Option
import pandas as pd
from datetime import datetime
import numpy as np
# from trade.helpers.Configuration import Configuration
from trade.helpers.Configuration import ConfigProxy
Configuration = ConfigProxy()
from trade.helpers.Context import Context
from trade.assets.Option import Option
from dateutil.relativedelta import relativedelta
import yfinance as yf
from threading import Thread
from trade.helpers.helper import generate_option_tick_new, identify_interval

structures = ['CallVertical', 'PutVertical', 'SyntheticForward', 'RiskReversal', 'Strangle', 'Straddle']

##Add Naked Call, Naked Put
def validate_leg(leg):
    assert isinstance(leg, list), "Leg must be a list of dictionaries"
    for opt in leg:
        assert isinstance(opt, dict), "Each option in a leg must be a dictionary"
        required_keys = {"strike", "expiration", "underlier", "right"}
        assert required_keys.issubset(opt.keys()), f"Option must have keys {required_keys}"
        assert isinstance(opt["strike"], (int, float)), "Strike must be a number"
        assert isinstance(opt["expiration"], str), "Expiration must be a str object"
        assert opt["right"] in {"c", "p"}, "Right must be 'c' (call) or 'p' (put)"


def EnforceSampleShape(sample):
    assert isinstance(sample, dict), "Sample must be a dictionary"
    assert any(x in ['long', 'short'] for x in sample.keys()), "Sample must have 'long' and 'short' keys"
    for key in ["long", "short"]:
        if key in sample:
            validate_leg(sample[key])


def all_same(attr, sample):
    """Check if a given attribute (e.g., 'right', 'expiration') is the same for both legs."""
    long_vals = {opt[attr] for opt in sample["long"]}
    short_vals = {opt[attr] for opt in sample["short"]}
    return long_vals == short_vals

def all_same_within(attr, sample, leg):
    """Check if a given attribute is the same within each leg."""
    vals = {opt[attr] for opt in sample[leg]}
    return len(vals) == 1

def only_one_underlier(sample):
    """Check if there is only one underlier in the sample."""
    undl = []
    for leg in sample.values():
        for opt in leg:
            undl.append(opt["underlier"])
    underliers = set(undl)
    return len(underliers) == 1

def all_different(attr, sample):
    """Check if a given attribute differs between the two legs."""
    long_vals = {opt[attr] for opt in sample["long"]}
    short_vals = {opt[attr] for opt in sample["short"]}
    return not long_vals.intersection(short_vals)

def extract_underlier(sample):
    assert only_one_underlier(sample), "Only one underlier is allowed"
    for leg in sample.values():
        for opt in leg:
            return opt["underlier"]




def SameRight(sample):

    EnforceSampleShape(sample)
    return all_same("right", sample)

def SameExpiration(sample):
    EnforceSampleShape(sample)
    return all_same("expiration", sample)

def SameExpirtationWithin(sample, leg):
    EnforceSampleShape(sample)
    return all_same_within("expiration", sample, leg)

def SameStrikeWithin(sample, leg):
    EnforceSampleShape(sample)
    return all_same_within("strike", sample, leg)

def SameStrike(sample):
    EnforceSampleShape(sample)
    return all_same("strike", sample)

def LongAndShort(sample):
    EnforceSampleShape(sample)
    if 'long' not in sample or 'short' not in sample:
        return False
    
    else:
        return True

def isOnlyCalls(sample):
    EnforceSampleShape(sample)
    if not LongAndShort(sample):
        return False
    key = list(sample.keys())[0]
    return all_same("right", sample) and sample[key][0]["right"].lower() == "c"

def isOnlyPuts(sample):
    EnforceSampleShape(sample)
    if not LongAndShort(sample):
        return False
    
    key = list(sample.keys())[0]
    return all_same("right", sample) and sample[key][0]["right"].lower() == "p"

def OnePerLeg(sample):
    
    """
    Check if there is only one leg per side
    """
    EnforceSampleShape(sample)

    if not LongAndShort(sample):
        return False

    if len(sample['long']) == 1 and len(sample['short']) == 1:
        return True
    
    else:
        return False
    
def OneWithinLeg(sample, leg):
    """
    Check if there is only one option within each leg
    """
    EnforceSampleShape(sample)
    if len(sample[leg]) == 1:
        return True
    
    else:
        return False
    
def SamePutCallWithin(sample, leg):
    """
    Check if all options within a leg are the same right
    """
    EnforceSampleShape(sample)
    if all_same_within("right", sample, leg):
        return True
    
    else:
        return False
    
def is_OneCallOnePut(sample):
    """
    Check if one side is ONLY one right and the other side is ONLY the other right
    """
    EnforceSampleShape(sample)

    if not isOnlyCalls(sample) and not isOnlyPuts(sample):
        return True
    
    else:
        return False


def is_CallVertical(sample):
    if not LongAndShort(sample):
        return False

    if (SameRight(sample) 
        and SameExpiration(sample) 
        and not SameStrike(sample) 
        and isOnlyCalls(sample) 
        and OnePerLeg(sample) 
        and LongAndShort(sample)):
            return True
    
    else:
            return False


def is_PutVertical(sample):
    if not LongAndShort(sample):
        return False
    
    if (SameRight(sample) 
        and SameExpiration(sample) 
        and not SameStrike(sample) 
        and isOnlyPuts(sample) 
        and OnePerLeg(sample) 
        and LongAndShort(sample)):
            return True
    
    else:
            return False

    
def is_SyntheticForward(sample):
    if not LongAndShort(sample):
        return False

    if (LongAndShort(sample) 
        and SameExpiration(sample) 
        and SameStrike(sample) 
        and is_OneCallOnePut(sample)
        and OnePerLeg(sample)):

        return True
    
    else:
        return False
    
def is_RiskReversal(sample):
    if not LongAndShort(sample):
        return False

    if (LongAndShort(sample) 
        and SameExpiration(sample) 
        and not SameStrike(sample) 
        and is_OneCallOnePut(sample)
        and OnePerLeg(sample)):

        return True
    
    else:
        return False
    
    
def is_Strangle(sample):
    if LongAndShort(sample):
        return False
    
    leg = list(sample.keys())[0]
    
    if  (SameExpirtationWithin(sample, leg) 
            and not SameStrikeWithin(sample, leg)
            and not OneWithinLeg(sample, leg)
            and not SamePutCallWithin(sample, leg)):
        return True
    
    else:
        return False
    
def is_Straddle(sample):
    if LongAndShort(sample):
        return False
    
    leg = list(sample.keys())[0]

    if  (SameExpirtationWithin(sample, leg) 
            and SameStrikeWithin(sample, leg)
            and not OneWithinLeg(sample, leg)
            and not SamePutCallWithin(sample, leg)):
        return True
    
    else:
        return False

def unKnownStructure(sample):
    if ( is_CallVertical(sample) or 
         is_PutVertical(sample) or 
         is_SyntheticForward(sample) or 
         is_RiskReversal(sample) or 
         is_Strangle(sample) or 
         is_Straddle(sample)):

        return False
    
    else:
        return True
    

stuctures_func = dict(zip(structures, [is_CallVertical, is_PutVertical, is_SyntheticForward, is_RiskReversal, is_Strangle, is_Straddle]))

def StructurePicker(sample):
    global stuctures_func
    for structure, func in stuctures_func.items():
        if func(sample):
            return structure
        
    return 'Unknown'


def PatchedOptionFunc( func_name: str, long_leg = [], short_leg = [], return_all = False, *args, **kwargs):
    """
    
    """
    acceptable_funcs = ['greeks', 'vol', 'spot']
    assert func_name in acceptable_funcs, f"Function {func_name} not in {acceptable_funcs}"
    structure_dict = {'long': [], 'short': []}

    def get_func_values(leg, leg_name, *args, **kwargs):
        values = []
        
        for l in leg: 
            assert isinstance(l, Option), "Leg must be an Option object"
            if leg_name == 'long':
                values.append(getattr(l, func_name)(*args, **kwargs))
            else:
                values.append(-getattr(l, func_name)(*args, **kwargs))
        
        structure_dict[leg_name] = values
    
    long_leg_thread = Thread(target=get_func_values, args=(long_leg, 'long', *args), kwargs=kwargs, name = f'{func_name}_long')
    short_leg_thread = Thread(target=get_func_values, args=(short_leg, 'short', *args), kwargs=kwargs, name = f'{func_name}_long')
    long_leg_thread.start()
    short_leg_thread.start()
    long_leg_thread.join(timeout=2*60)
    short_leg_thread.join(timeout=2*60)
    structure_dict['total'] = sum(structure_dict['long']) + sum(structure_dict['short'])


    if return_all:
        return structure_dict
    else:
        return structure_dict['total']


    
class OptionStructure:
    ## Will not be implementing singleton

    def __init__(self, structure, **kwargs):

        """
        Sample Strucutre:
            sample = {
                'long': [{'strike': 175.0, 'expiration': '2025-03-21', 'underlier': 'AAPL', 'right': 'c'},],
                'short': [{'strike': 200.0, 'expiration': '2025-03-21', 'underlier': 'AAPL', 'right': 'p'},],
            }
        
        """
        assert isinstance(structure, dict), "Structure must be a dictionary"
        assert only_one_underlier(structure), "Only one underlier is allowed"
        self.base_structure = structure
        self.__long = []
        self.__short = []
        self.direction = None
        self.Structure = {}  
        self.StructureName = StructurePicker(structure)

        today = datetime.today()
        start_date_date = today - relativedelta(years = 4)
        start_date = datetime.strftime(start_date_date, format='%Y-%m-%d')
        end_date = datetime.strftime(today, format='%Y-%m-%d')
        self.__ticker = extract_underlier(structure)
        self.__security = yf.Ticker(self.__ticker.upper())
        self.timewidth = Configuration.timewidth or '1'
        self.timeframe = Configuration.timeframe or 'day'
        self.__start_date = Configuration.start_date or start_date
        self.__end_date = Configuration.end_date or end_date
        self.default_fill = kwargs.get('default_fill', 'midpoint')
        self.__sigma = None
        self.__pv = None
        self.run_chain = kwargs.get('run_chain', False)
        self._init_structure()
        self.pv_set_thread = Thread(target=self.__set_pv, name = f'{self.ticker}_PV_Setter')
        self.sigma_set_thread = Thread(target=self.__set_sigma, name = f'{self.ticker}_Sigma_Setter')
        self.pv_set_thread.start()
        self.sigma_set_thread.start()


        

    @property
    def ticker(self):
        return self.__ticker
    
    @property
    def security(self):
        return self.__security

    @property
    def start_date(self):
        return self.__start_date
    
    @property
    def end_date(self):
        return self.__end_date
    
    @property
    def long(self):
        return self.__long
    
    @property
    def short(self):
        return self.__short
    
    @property
    def sigma(self):
        if not self.is_sigma_set():
            print("Sigma not set")
        return self.__sigma
    
    @property
    def pv(self):
        if not self.is_pv_set():
            print("PV not set")
        return self.__pv

    @long.setter
    def long(self, value):
        self.__long = value

    @short.setter
    def short(self, value):
        self.__short = value
    
        
    def __str__(self):
        return f'{self.StructureName}({self.__ticker}, Build On: {self.__end_date})'
    
    def __repr__(self):
        return f'{self.StructureName}({self.__ticker}, Build On: {self.__end_date})'
    
    def _init_structure(self):
        for direction in self.base_structure:
            legs_list = []
            legs_positions = self.base_structure[direction]
            for pos in legs_positions:
                legs_list.append(Option(pos['underlier'], pos['strike'], pos['expiration'], pos['right'], run_chain = self.run_chain, default_fill=self.default_fill))
                self.asset = legs_list[0].asset
            setattr(self, direction, legs_list)
            self.Structure[direction] = legs_list
    
    def __set_pv(self):
        pv_long, pv_short = 0, 0
        if self.long is not None:
            for leg in self.long:
                pv_long += leg.pv

        if self.short is not None:
            for leg in self.short:
                pv_short += leg.pv
        

        self.__pv =  pv_long - pv_short

    

    def __set_sigma(self):
        sigma_long, sigma_short = 0, 0
        if self.long is not None:
            for leg in self.long:
                sigma_long += leg.sigma

        if self.short is not None:
            for leg in self.short:
                sigma_short += leg.sigma
            
        self.__sigma = sigma_long - sigma_short

    def is_pv_set(self):
        return self.__pv is not None
    
    def is_sigma_set(self):
        return self.__sigma is not None

    def greeks(self,
               greek_type = 'greek', 
                ts_start = None, 
                ts_end = None, 
                ts_timewidth = None, 
                ts_timeframe = None,
                return_all = False):
        """
        The greeks method returns a timeseries dataframe for greeks based. Only available for BSM model

        PARAMS
        ______
        ts (Bool): True to return dataframe timeseries, false to return spot in a dict
        ts_start (str|datetime): Start Date
        ts_end (str|datetime): End Date
        ts_timewidth (str|int): Steps in timeframe
        ts_timeframe (str): Target timeframe for series 
        greek_type (str): Type of greek to return. Default is 'greek'.
            'greek' returns all greek, while passing 'delta', 'gamma', 'theta', 'vega' returns only the specific greek
        return_all (bool): True to return all from each leg, False to return only the aggregate greeks
        

        RETURNS
        _________
        pd.DataFrame or dict
        """
        if ts_timeframe is None:
            ts_timeframe = 'day'
        if ts_timewidth is None:
            ts_timewidth = self.timewidth
        if ts_start is None:
            ts_start = self.start_date
        if ts_end is None:
            ts_end = pd.to_datetime(self.end_date ) + relativedelta(days = 1)
        return PatchedOptionFunc(
                'greeks',
                long_leg = self.long,
                short_leg = self.short,
                return_all = return_all,
                greek_type = greek_type,
                ts_start = ts_start,
                ts_end = ts_end,
                ts_timewidth = ts_timewidth,
                ts_timeframe = ts_timeframe)
    

    def spot(self, 
    ts = False, 
    ts_start = None, 
    ts_end = None, 
    ts_timewidth = None, 
    ts_timeframe = None,
    spot_type = 'quote',
    return_all = False):
        """
        The spot method returns a dataframe for latest quote price or a dictionary for last available close. 

        PARAMS
        ______
        ts (Bool): True to return dataframe timeseries, false to return spot in a dict
        ts_start (str|datetime): Start Date
        ts_end (str|datetime): End Date
        ts_timewidth (str|int): Steps in timeframe
        ts_timeframe (str): Target timeframe for series 
        return_all (bool): True to return all from each leg, False to return only the aggregate greeks
        
        RETURNS
        _________
        pd.DataFrame
        """

        return PatchedOptionFunc(
                'spot',
                ts = ts,
                long_leg = self.long,
                short_leg = self.short,
                return_all = return_all,
                ts_start = ts_start,
                ts_end = ts_end,
                spot_type = spot_type,
                ts_timewidth = ts_timewidth,
                ts_timeframe = ts_timeframe)
    
    def vol(self, 
        ts = False, 
        ts_start = None, 
        ts_end = None, 
        ts_timewidth = None, 
        ts_timeframe = None, 
        return_all = None):
            """
            The vol method returns a dataframe for latest quote price vol
            If ts is set to true, It returns the timeseries of vol based on the model.
            No current vol available for mcs model

            PARAMS
            ______
            ts (Bool): True to return dataframe timeseries, false to return spot in a dict
            ts_start (str|datetime): Start Date
            ts_end (str|datetime): End Date
            ts_timewidth (str|int): Steps in timeframe
            ts_timeframe (str): Target timeframe for series 
            return_all (bool): True to return all from each leg, False to return only the aggregate greeks
            

            RETURNS
            _________
            pd.DataFrame
            """

            return PatchedOptionFunc(
                    'vol',
                    ts = ts,
                    long_leg = self.long,
                    short_leg = self.short,
                    return_all = return_all,
                    ts_start = ts_start,
                    ts_end = ts_end,
                    ts_timewidth = ts_timewidth,
                    ts_timeframe = ts_timeframe)



