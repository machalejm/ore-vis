import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ipywidgets
from ORE import parsePeriod, Parameters, OREApp
from math import ceil
import re
import utilities

############################ WORKAROUND ###################################

def market_object_type_id_workarounds(object_type, object_id):
    """This function changes the object types and ids to an earlier version for backwards compatibility."""
    
    if '/' in object_id:
        return [object_type, object_id]
    else:
        if object_type == 'yieldCurve':
            if object_id.startswith('RIC:') or object_id.startswith('BBG:') or object_id.startswith('FIGI:'):
                return ['equityCurve', f"Equity/CCY/{object_id}"]
            else:
                return [object_type, f"Yield/{object_id.split('-')[0]}/{object_id}"]
        elif object_type == 'inflationCuve':
            return ['inflationCurve', f"Inflation/{object_id.split('_')[0]}/{object_id}"]
        elif object_type == 'eqVol':
            return [object_type, f"EquityVolatility/CCY/{object_id}"]
        elif object_type == 'fxVol':
            return [object_type, f"FXVolatility/{object_id[:3]}/{object_id[3:]}/{object_id}"]
        elif object_type == 'commodityCuve':
            return ['commodityCurve', f"Commodity/CCY/{object_id}"]
        elif object_type == 'commVol':
            return [object_type, f"CommodityVolatility/CCY/{object_id}"]
        else:
            return [object_type, object_id]

########################## DELTA STRIKES ##################################

class DeltaStrike:
    def __init__(self, styled_strike=None, call_strike=None):
        """Need to provide either styled_strike like 30P or a pseudo-"call delta" eg. 80"""
        if styled_strike is not None:
            regex_pattern = r"^(\d{1,2})([CP])|ATM$"
            regex_match = re.search(regex_pattern, styled_strike)
            if regex_match is None:
                raise ValueError(f"Unhandled strike! {styled_strike}")
            else:
                if styled_strike == 'ATM':
                    self.strike = 50
                elif regex_match.group(2) == 'C':
                    self.strike = int(regex_match.group(1))
                else:
                    self.strike = 100 - int(regex_match.group(1))
        elif call_strike is not None:
            self.strike = call_strike
        else:
            raise ValueError("Not enough data provided to Delta Strike constructor.")

    def __str__(self):
        if self.strike == 50:
            return 'ATM'
        elif self.strike < 50:
            return f"{self.strike}C"
        else:
            return f"{100 - self.strike}P"

    def __lt__(self, other):
        return self.strike < other.strike

    def __gt__(self, other):
        return self.strike > other.strike

    def __eq__(self, other):
        return self.strike == other.strike

def delta_strike_transform(k):
    return DeltaStrike(k).strike

def delta_strike_axis_labels():
    strike_values = np.arange(0, 120, 10)
    strike_labels = [str(DeltaStrike(call_strike=s)) for s in strike_values]
    return strike_values, strike_labels

################## USEFUL FUNCTIONS & CONSTANTS ######################################

def parse_bool(s):
    lookup = {"Y": True, "YES": True, "TRUE": True, "True": True, "true": True, "1": True,
              "N": False, "NO": False, "FALSE": False, "False": False, "false": False, "0": False}
    return lookup[s]

def log_transform(t):
    return np.log(t + 0.25)  # log

def scale_time(t):
    return log_transform(t)

def add_scaled_time(data):
    data['f(T)'] = data['time'].astype(float).apply(lambda x: scale_time(x))
    return data

months = [1, 3, 6]
years = [1, 2, 5, 10, 30, 50]
tick_time = np.array(list(map(float, [i/12 for i in months] + years)))
tick_f_time = scale_time(tick_time)
tick_labels = [f'{m}M' for m in months] + [f'{y}Y' for y in years]

object_type_labels = {'fxVol': "FX Vols", 'equityCurve': "Equity Curves", 'eqVol': "Equity Vols", 'irVol': "Rates Vols", 
                      "yieldCurve": "Yield Curves", "inflationCurve": "Inflation Curves", 'commodityCurve': "Commodity Curves", 'commVol': "Commodity Vols"}

########################## VOLATILITY STUFF ##############################

arbitrage_colour_scheme = {
    'No Arbitrage': 'green',
    'Butterfly': 'red',
    'Call Spread': 'orange',
    'Butterfly + Call Spread': '#8B0000',
    'Calendar': 'blue',
    'Butterfly + Calendar': '#00008B',
    'Call Spread + Calendar': '#8B008B', 
    'All Types': 'black'
}
# #8B0000=DarkGreen, #00008B = DarkBlue, #8B008B = DarkMagenta
# https://www.w3schools.com/cssref/css_colors.asp

option_type_mappings = {
    "swaption": ["SwaptionVolatility", "atm_spread", "data_strike_spread_grid", float],
    "capfloor": ["CapFloorVolatility", "absolute_strike", "data_strike_grid", float],
    "fxoption": ["FXVolatility", "delta_strike", "data_delta_grid", delta_strike_transform],
    "eqoption": ["EquityVolatility", "moneyness", "data_moneyness_grid", float],
    "commoption": ["CommodityVolatility", "moneyness", "data_delta_grid", delta_strike_transform],
}

def add_summed_arbitrage(data):
    data['arbitrage'] = 1 * data['butterflyArb'] + 2 * data['callSpreadArb']
    if 'calendarArb' in data.columns:
        data['arbitrage'] = data['arbitrage'] + 4 * data['calendarArb'].fillna(False)
    data['arbitrage'] = data['arbitrage'].map({i: j for i, j in enumerate(arbitrage_colour_scheme.keys())})
    
    return data

def vol_report(market_calibration_report, curve_id, option_type):
    supported_vol_types = ['swaption', 'capfloor', 'fxoption', 'eqoption', 'commoption']
    assert option_type.lower() in supported_vol_types, f'Supported volatility types: f{supported_vol_types}, does not contain {option_type}'
    curve_id_stem, strike_style, grid_label, strike_transformer = option_type_mappings[option_type.lower()]

    vols = market_calibration_report[market_calibration_report['MarketObjectId'].str.startswith(curve_id_stem)]
    vol = vols[vols['MarketObjectId'] == curve_id]
    assert len(vol) > 0, f"No data found id stem {curve_id_stem} and name {curve_id}."

    vol_static_info = vol[vol['ResultKey1']==""][['ResultId', 'ResultValue']]
    vol_expiry_grid = vol[vol['ResultId'] == 'expiry']
    vol_tenor_grid = vol[vol['ResultId'] == 'tenor']
    time_expiry_map = vol_expiry_grid.set_index('ResultKey1')['ResultValue'].to_dict()

    vol_data = vol[~vol.index.isin(
        vol_static_info.index.tolist() \
        + vol_expiry_grid.index.tolist() \
        + vol_tenor_grid.index.tolist()
    )].rename(columns={"MarketObjectId": "name", "ResultKey1": "time", "ResultKey2": strike_style, "ResultKey3": "underlying"}).copy()

    vol_data = pd.pivot(vol_data, 
             index=['MarketObjectType', 'name', 'time', 'underlying', strike_style],
             columns=['ResultId'],
             values=['ResultValue']
            )
    
    float_columns = ["time", 'forward', 'prob', 'strike', 'vol']
    period_columns = ["underlying"] if option_type in ('swaption', 'capfloor') else list()
    bool_columns = ['butterflyArb', 'callSpreadArb', 'calendarArb']
    # strike_columns = [strike_style]
    column_caster = {"double": (lambda x: float(x), float_columns), 
                     "period": (lambda x: parsePeriod(x), period_columns),
                     "bool": (lambda x: parse_bool(x), bool_columns),
                     # "strike": (lambda x: strike_transformer(x), strike_columns), 
                    }

    vol_data.columns = vol_data.columns.droplevel(0)
    vol_data = vol_data.reset_index().rename_axis(None, axis=1).sort_values(by=['time', 'underlying', strike_style])
    vol_data['expiry'] = pd.to_datetime(vol_data['time'].replace(time_expiry_map))
    vol_data['grid_style'] = grid_label

    for data_type, (type_parser, column_names) in column_caster.items():
        for c in (c for c in column_names if c in vol_data.keys()):
            vol_data[c] = vol_data[c].apply(type_parser)

    extra_static_data = pd.DataFrame(
        dict(zip(vol_static_info.columns, [['strikeStyle', 'curveId'], ['ATM spread'.replace('_', ' ').title(), curve_id]])))
    extended_static_data = pd.concat([vol_static_info, extra_static_data]).reset_index(drop=True)

    return vol_data

def swaption_vol_report(market_calibration_report, curve_id):
    return vol_report(market_calibration_report, curve_id, option_type='swaption')

def capfloor_vol_report(market_calibration_report, curve_id):
    return vol_report(market_calibration_report, curve_id, option_type='capfloor')

def fxoption_vol_report(market_calibration_report, curve_id):
    return vol_report(market_calibration_report, curve_id, option_type='fxoption')

def eqoption_vol_report(market_calibration_report, curve_id):
    return vol_report(market_calibration_report, curve_id, option_type='eqoption')

def commoption_vol_report(market_calibration_report, curve_id):
    return vol_report(market_calibration_report, curve_id, option_type='commoption')

def _vol_plotter(vol_data, option_type, abs_strikes_bool=False, arbitrage=True):
    supported_vol_types = ['swaption', 'capfloor', 'fxoption', 'eqoption', 'commoption']
    assert option_type.lower() in supported_vol_types, f'Supported volatility types: {supported_vol_types} does not contain {option_type}.'
    curve_id_stem, strike_column, grid_style, strike_transformer = option_type_mappings[option_type.lower()]

    data = vol_data.copy()
    name = data['name'].iloc[0]
    data = add_summed_arbitrage(vol_data)
    data = add_scaled_time(vol_data)
    name_grid_vols = data[(data['name'] == name)]
    assert not len(name_grid_vols) == 0, "no data in swaption vol"
    max_vol = max(name_grid_vols['vol'].astype(float))

    fig = go.Figure()

    for arb, color in (arbitrage_colour_scheme.items() if arbitrage else [[None, 'green'], ]):
        # Select the subset and get the data
        name_grid_arb_vols = data[
            (data['name'] == name) &
            (data['grid_style'] == grid_style) &
            (data['arbitrage'] == (arb or data['arbitrage']))].copy()

        times, f_times, vols = name_grid_arb_vols[['time', 'f(T)', 'vol']].astype(float).to_numpy().T
        underlyings = None
        if 'underlying' in name_grid_arb_vols and not all(name_grid_arb_vols['underlying']==""):
            underlyings = name_grid_arb_vols['underlying']
        formatted_strikes = name_grid_arb_vols[strike_column]
        strikes = formatted_strikes.apply(strike_transformer)
        abs_strikes = name_grid_arb_vols['strike'].astype(float)

        # This is ugly because the underlying field may or may not be there
        hover_texts = list()
        hover_texts.append("T = " + pd.Series(times).astype(str))
        if underlyings is not None: hover_texts.append("Und = " + pd.Series(underlyings).astype(str))
        hover_texts.append(f"{strike_column} = " + pd.Series(formatted_strikes).astype(str))
        if strike_column != 'strike': hover_texts.append(f"K = " + name_grid_arb_vols['strike'].astype(str))
        hover_texts.append("vol = " + pd.Series(vols).astype(str))
        hover_texts.append("Abritrage: " + name_grid_arb_vols['arbitrage'])
        # next line just concatenates the above together and slaps an arbitage label on the end.
        hover_texts = pd.Series(zip(*hover_texts)).apply('<br>'.join).astype(str)
        hover_texts = hover_texts.values

        plot_strikes = abs_strikes if abs_strikes_bool else strikes
        
        if underlyings is not None:
            for underlying in sorted(underlyings.unique().tolist()):
                scatter_points = go.Scatter3d(
                    x=plot_strikes[underlyings == underlying],
                    y=f_times[underlyings == underlying],
                    z=vols[underlyings == underlying],
                    hoverinfo='text',
                    hovertext=hover_texts[underlyings == underlying],
                    mode='markers', marker=dict(size=4, opacity=0.5, color=color),
                    showlegend=True, name=arb or 'Volatility',
                    legendgroup=str(underlying), legendgrouptitle={'text': str(underlying)}
                    )
                fig.add_trace(scatter_points)
        else:
            scatter_points = go.Scatter3d(
                x=plot_strikes,
                y=f_times,
                z=vols,
                hoverinfo='text',
                hovertext=hover_texts,
                mode='markers', marker=dict(size=4, opacity=0.5, color=color),
                showlegend=True, name=arb or 'Volatility'
                )
            fig.add_trace(scatter_points)

    title = f"{name} {grid_style.replace('_', ' ')} volatility surface"

    xaxis_formatting = {'title': f"Strike"}
    if strike_column == 'delta_strike' and abs_strikes_bool == False:
        xaxis_values, xaxis_labels = delta_strike_axis_labels()
        xaxis_formatting['range'] = [-5, 105]
        xaxis_formatting['tickvals'] = xaxis_values
        xaxis_formatting['ticktext'] = xaxis_labels

    template = ["plotly", "plotly_white", "plotly_dark", 
                "ggplot2", "seaborn", "simple_white", "none"][1]
    fig.update_layout(
        template=template,
        autosize=False, width=800, height=800,
        title=title,
        scene=dict(
            xaxis=xaxis_formatting,
            yaxis=dict(title='Time', tick0=0.0, ticktext=tick_labels, tickvals=tick_f_time),
            zaxis=dict(title='Volatility', ticks='outside', range=[0, ceil(max_vol * 50.0) / 50.0]),
            zaxis_tickformat='.2%'
        ),
        margin=dict(pad=200)
    )

    return fig

def vol_plotter(market_calibration_report, curve_id, abs_strikes_bool=False, arbitrage=True):
    mappings = {v[0]: k for k,v in option_type_mappings.items()}
    option_type = mappings[curve_id.split('/')[0]]
    
    if option_type == "swaption":
        vr = swaption_vol_report(market_calibration_report, curve_id)
    elif option_type == "capfloor":
        vr = capfloor_vol_report(market_calibration_report, curve_id)
    elif option_type == "fxoption":
        vr = fxoption_vol_report(market_calibration_report, curve_id)
    elif option_type == "eqoption":
        vr = eqoption_vol_report(market_calibration_report, curve_id)
    elif option_type == "commoption":
        vr = commoption_vol_report(market_calibration_report, curve_id)
    else:
        vr = None
    return _vol_plotter(vr, option_type, abs_strikes_bool=abs_strikes_bool, arbitrage=arbitrage)

################## CURVE STUFF #######################################

curve_type_mappings = {
    "yield": ["Yield", None, None, None],
    "equity": ["Equity", None, None, None],
    "commodity": ["Commodity", None, None, None],
    "inflation": ["Inflation", None, None, None],
}

def curve_report(market_calibration_report, curve_id_token, curve_type):
    supported_curve_types = ['yield', 'inflation', 'equity', 'commodity']
    assert curve_type.lower() in supported_curve_types, f'Supported curve types: {supported_curve_types}, not {curve_type}.'
    curve_id_stem, _, _, _ = curve_type_mappings[curve_type.lower()]

    curves = market_calibration_report[market_calibration_report['MarketObjectId'].str.startswith(curve_id_stem + '/')]
    curve = curves[curves['MarketObjectId'].str.split('/').apply(lambda x: curve_id_token in x)]
    assert len(curve) > 0, f"No data found with stem {curve_id_stem} and at least one token {curve_id_token}."

    curve_static_info = curve[curve['ResultKey1']==""][['ResultId', 'ResultValue']]
    curve_maturity_grid = np.sort(curve[curve['ResultKey1'] != '']['ResultKey1'].unique())
    time_maturity_map = curve[curve['ResultId']=='time'].set_index('ResultKey1')['ResultValue'].to_dict()

    curve_data = curve[~curve.index.isin(
        curve_static_info.index.tolist()
    )].rename(columns={"MarketObjectId": "name", "ResultKey1": "maturity"}).copy()
    
    curve_data = pd.pivot(curve_data.drop(['ResultKey2', 'ResultKey3'], axis=1), 
             index=['MarketObjectType', 'name', 'maturity'],
             columns=['ResultId'],
             values=['ResultValue']
            ).sort_index()

    float_columns = ["time", 'zeroRate', 'discountFactor', 'cpi', 'price']
    column_caster = {"double": (lambda x: float(x), float_columns), }

    curve_data.columns = curve_data.columns.droplevel(0)
    curve_data = curve_data.reset_index().rename_axis(None, axis=1)
    curve_data['maturity'] = pd.to_datetime(curve_data['maturity']) #.replace(time_maturity_map))

    for data_type, (type_parser, column_names) in column_caster.items():
        for c in (c for c in column_names if c in curve_data.keys()):
            curve_data[c] = curve_data[c].apply(type_parser)

    return curve_data

def _curve_plotter(curves_data, curve_type, _2=None, _3=None):
    supported_curve_types = ['yield', 'inflation', 'equity', 'commodity']
    assert curve_type.lower() in supported_curve_types, f'Supported curve types: {supported_curve_types}, not {curve_type}.'
    
    curves_data = add_scaled_time(curves_data)
    
    fig = go.Figure()
    for name in curves_data['name'].unique():
        curve_data = curves_data[curves_data['name'] == name].sort_values(by=['time'])
        hover_texts = list()
        hover_texts.append(curve_data['name'].astype(str))
        hover_texts.append("Maturity = " + curve_data['maturity'].astype(str))
        hover_texts.append("T = " + (curve_data['time']*365).astype(str))
        if not curve_type in ['Commodity']:
            hover_texts.append("Zero Rate = " + (curve_data['zeroRate']*100.0).map("{:.4f}%".format))
        else:
            hover_texts.append("Price = " + (curve_data['price']).astype(str))
        if curve_type in ['Yield']:
            hover_texts.append("Discount = " + curve_data['discountFactor'].astype(str))
        if curve_type in ['Inflation']:
            hover_texts.append("CPI = " + curve_data['cpi'].astype(str))
        # next line just concatenates the above together.
        hover_texts = pd.Series(zip(*hover_texts)).apply('<br>'.join).astype(str).values
        x = curve_data['time']
        x = curve_data['f(T)']
        y = curve_data['zeroRate'] if not curve_type in ['Commodity'] else curve_data['price']
        fig.add_trace(
            go.Scatter(
                x=x, y=y, 
                mode='lines+markers', name=name, line=dict(shape='linear'),
                hoverinfo='text', hovertext=hover_texts
            )
        )
    template = ["plotly", "plotly_white", "plotly_dark", 
                "ggplot2", "seaborn", "simple_white", "none"][1]
    yaxis_settings = dict()

    if not curve_type in ['Commodity']:
        yaxis_settings['tickformat'] = '.2%'
    fig.update_layout(template=template, xaxis_showgrid=False, yaxis_showgrid=False,
                      xaxis=dict(title='Time', tick0=0.0, ticktext=tick_labels, tickvals=tick_f_time),
                     yaxis=yaxis_settings)

    return fig

def yield_curve_plotter(market_calibration_report, curve_id, _1=None, _2=None):
    curve_type = 'Yield'
    curve_name = curve_id.split('/')[-1]
    
    cr = curve_report(market_calibration_report, curve_name, curve_type)
    return _curve_plotter(cr, curve_type)

def equity_curve_plotter(market_calibration_report, curve_id, _1=None, _2=None):
    curve_type = 'Equity'
    curve_name = curve_id.split('/')[-1]
    
    cr = curve_report(market_calibration_report, curve_name, curve_type)
    return _curve_plotter(cr, curve_type)

def commodity_curve_plotter(market_calibration_report, curve_id, _1=None, _2=None):
    curve_type = 'Commodity'
    curve_name = curve_id.split('/')[-1]
    
    cr = curve_report(market_calibration_report, curve_name, curve_type)
    return _curve_plotter(cr, curve_type)

def inflation_curve_plotter(market_calibration_report, curve_id, _1=None, _2=None):
    curve_type = 'Inflation'
    curve_name = curve_id.split('/')[-1]
    
    cr = curve_report(market_calibration_report, curve_name, curve_type)
    return _curve_plotter(cr, curve_type)

################## WIDGET STUFF #######################################

class CurvePlotterWidgets():

    def __init__(self, market_calibration_report):
        # We copy the report to the class, clean up the ids to be backwards compatible, also save the summary
        self.market_calibration_report = market_calibration_report.copy()
        # apply the workaround per row with a lambda
        row_workaround_fxn = lambda mcr_row: market_object_type_id_workarounds(mcr_row['MarketObjectType'], mcr_row['MarketObjectId'])
        self.market_calibration_report[['MarketObjectType', 'MarketObjectId']] = \
            pd.DataFrame(self.market_calibration_report.apply(row_workaround_fxn, axis=1).tolist())
        # unique ids per type
        self.market_calibration_report_objects = {
            k: v['MarketObjectId'].tolist() 
            for k, v in self.market_calibration_report[['MarketObjectType', 'MarketObjectId']].drop_duplicates().groupby('MarketObjectType', )
        }

        default_object_type = 'yieldCurve'
        default_object = 'USD'
        
        # Select type of object e.g. yield curves, fx vols...
        object_type_selector_options = [(object_type_labels.get(x, x), x) for x in self.market_calibration_report_objects.keys()]
        self.object_type_selector = ipywidgets.Dropdown(options=object_type_selector_options, value=default_object_type)
    
        # Select object e.g, EURUSD vol surface
        object_selector_options = \
            sorted({(y,y) 
                    for x in self.market_calibration_report_objects[default_object_type] 
                    for y in x.split('/')[1:]})
        self.object_selector = ipywidgets.Dropdown(options=object_selector_options, value=default_object)
    
        # Absolute strike toggle
        self.abs_strike_checkbox = ipywidgets.Checkbox(value=True, description="Absolute Strikes", disabled=True)
        
        # Show Arbitrage toggle
        self.arbitrage_checkbox = ipywidgets.Checkbox(value=False, description="Show Arbitrage", disabled=True)

        # Will hold the graphs
        self.visualization_frame = ipywidgets.Output(layout={'border': '1px'})

        # The full collection of widgets + graph
        self.output_window = ipywidgets.VBox([
            ipywidgets.HBox(
                [
                    ipywidgets.VBox([self.object_type_selector, self.object_selector]),
                    ipywidgets.VBox([self.abs_strike_checkbox, self.arbitrage_checkbox]) 
                ]),
            self.visualization_frame
        ])
        
        # Changes to widgets should kick off updates and plots
        self.object_type_selector.observe(self.__update_object_selector_options, 'value')
        self.object_selector.observe(self.__plot, 'value')
        self.abs_strike_checkbox.observe(self.__plot, 'value')
        self.arbitrage_checkbox.observe(self.__plot, 'value')

        self.__plot()

    def __plot(self, *args):
        """Read values from widgets and use them to make a plot in the visualization frame."""
        
        objects_type = self.object_type_selector.value
        object_ = self.object_selector.value
        abs_strikes = self.abs_strike_checkbox.value
        arbitrage = self.arbitrage_checkbox.value
        
        with self.visualization_frame:
            self.visualization_frame.clear_output()
            fig = self.plotters[objects_type](self.market_calibration_report, object_, abs_strikes, arbitrage)
            if fig is not None:
                fig.show()

    def __update_object_selector_options(self, *args):
        objects_type = self.object_type_selector.value
        
        if objects_type in ('yieldCurve'):
            self.abs_strike_checkbox.disabled = True
            self.arbitrage_checkbox.disabled = True
            self.object_selector.options = \
                sorted({(y,y) for x in self.market_calibration_report_objects['yieldCurve'] for y in x.split('/')[1:]})
        elif objects_type in ('equityCurve', ):
            self.abs_strike_checkbox.disabled = True
            self.arbitrage_checkbox.disabled = True
            self.object_selector.options = \
                sorted({(x.split('/')[2], x.split('/')[2]) for x in self.market_calibration_report_objects['equityCurve']})
        elif objects_type in ('inflationCurve', ):
            self.abs_strike_checkbox.disabled = True
            self.arbitrage_checkbox.disabled = True
            self.object_selector.options = \
                sorted({(x.split('/')[1], x.split('/')[1]) for x in self.market_calibration_report_objects['inflationCurve']})
        else:
            self.abs_strike_checkbox.disabled = False
            self.arbitrage_checkbox.disabled = False
            self.object_selector.options = self.market_calibration_report_objects[objects_type]
    
    # data
    market_calibration_report = None
    market_calibration_report_objects = None
    
    # widgets
    object_type_selector = None
    object_selector = None
    abs_strike_checkbox = None
    arbitrage_checkbox = None
    visualization_frame = None
    output_window = None

    # pointers to functions
    plotters = {'irVol': vol_plotter, 'commodityCurve': commodity_curve_plotter, 'yieldCurve': yield_curve_plotter, 
                'inflationCurve': inflation_curve_plotter, 'fxVol': vol_plotter, 'eqVol': vol_plotter, 
                'commVol': vol_plotter, 'equityCurve': equity_curve_plotter}

################## ORE RUNNER #######################################

def run_ore_generate_market_calibration_report(input_directory):
    params = Parameters()
    params.fromFile(f"{input_directory}/ore.xml")
    ore = OREApp(params)
    ore.run()
    reports = {report_name: ore.getReport(report_name) for report_name in ore.getReportNames()}
    return utilities.format_report(reports['todaysmarketcalibration'])