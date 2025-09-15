import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State
from core_calculations import *

# Default solar observation parameters
DEFAULT_LAT, DEFAULT_LON, DEFAULT_TZ_OFF = 51.49, -0.177, 0
DEFAULT_TIME = dt.datetime(2025, 3, 20, 12, 0, 0)

# Build the Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="helio-plot"),

    # Heliostat selection
    html.Div([
        html.Label("Select Heliostats:"),
        dcc.Checklist(
            id='helio-select',
            options=[{'label': k, 'value': k} for k in H_POSITIONS],
            value=list(H_POSITIONS.keys()),
            inline=True
        )
    ], style={"margin":"10px 0"}),

    # Location input
    html.Div([
        html.Label("Latitude (deg):"),
        dcc.Input(id='lat-input', type='number', placeholder='(-) = S, (+) = N', value=DEFAULT_LAT),
        html.Label("Longitude (deg):"),
        dcc.Input(id='lon-input', type='number', placeholder='(-) = W, (+) = E', value=DEFAULT_LON),
        html.Label("TZ Offset (decimal hrs):"),
        dcc.Input(id='tz-input', type='number', placeholder='e.g.+5.75', value=DEFAULT_TZ_OFF)
    ], style={"margin": "15px 5"}),

    # Date-time input for solar calculation
    html.Div([
        html.Label("Enter date and time in local TZ:"),
        dcc.Input(
            id='datetime-input', type='text',
            placeholder='YYYY-MM-DD HH:MM:SS',
            value=DEFAULT_TIME.strftime('%Y-%m-%d %H:%M:%S')
        ),
        html.Div(id='elevation-display', style={"margin":"10px 0", "font-weight":"bold"})
    ]),

    # NEW: Source model + LED parameters
    html.Div([
        html.Label("Source model:"),
        dcc.RadioItems(
            id='source-model',
            options=[
                {'label': ' Sun (parallel rays)', 'value': 'sun'},
                {'label': ' LED point source (finite distance + divergence)', 'value': 'led'}
            ],
            value='sun',
            inline=True
        ),
        html.Div([
            html.Label("LED distance (mm):"),
            dcc.Input(id='led-distance-mm', type='number', value=9800.0, min=0,
                      step=10, style={"width":"10ch"}),
            html.Label("  Full divergence (deg):"),
            dcc.Input(id='led-divergence-deg', type='number', value=6.5, min=0, max=180, step=0.1, style={"width":"8ch"}),
        ], id='led-input-block', style={"margin":"8px 0"})
    ], style={"margin":"12px 0", "padding":"8px 6px", "border":"1px dashed #bbb", "borderRadius":"8px"}),

    # Slider containers for each heliostat
    html.Div([
        html.Div([
            html.Div(id=f"tilt-display-{i}", style={"margin":"10px 0","font-weight":"bold"}),
            dcc.Slider(id=f"inward-tilt-slider-{i}", min=0, max=80, step=0.1, value=0, marks=None, updatemode="drag"),
        ], id=f"slider-container-{i}")
        for i in range(1,5)
    ])
])

# Callback: enable/disable LED inputs based on model
@app.callback(
    Output('led-distance-mm','disabled'),
    Output('led-divergence-deg','disabled'),
    Input('source-model','value')
)
def toggle_led_inputs(model):
    disabled = (model != 'led')
    return disabled, disabled

# Callback: control slider visibility based on selection
@app.callback(
    [Output(f'slider-container-{i}','style') for i in range(1,5)],
    Input('helio-select','value')
)
def update_slider_visibility(selected):
    return [{'display':'block'} if f'H{i}' in selected else {'display':'none'} for i in range(1,5)]

# Callback: display elevation angle when datetime or location changes
@app.callback(
    Output('elevation-display','children'),
    Input('datetime-input','value'),
    Input('lat-input','value'),
    Input('lon-input','value'),
    Input('tz-input','value')
)
def update_elevation(datetime_str, lat, lon, tz_off):
    try:
        trial_local = dt.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return "Invalid datetime format; use YYYY-MM-DD HH:MM:SS"
    trial_utc = trial_local - dt.timedelta(hours=tz_off)
    alt, azi = solar_angles(lat, lon, trial_utc)
    return f"Elevation angle: {alt:.2f}°, Azimuth angle: {azi:.2f}°"

# Callback: update slider labels
@app.callback(
    [Output(f"tilt-display-{i}",  "children") for i in range(1,5)],
    [Input(f"inward-tilt-slider-{i}", "value") for i in range(1,5)]
)
def display_values(*vals):
    tilts = vals[:4]
    return [f"H{i} Inward tilt ϕ: {t}°" for i, t in enumerate(tilts, 1)]

# Callback: rebuild the 3D figure based on all inputs
@app.callback(
    Output("helio-plot", "figure"),
    Input('helio-select','value'),
    Input('datetime-input','value'),
    Input('lat-input','value'),
    Input('lon-input','value'),
    Input('tz-input','value'),
    Input('source-model','value'),               # ——— NEW
    Input('led-distance-mm','value'),            # ——— NEW
    Input('led-divergence-deg','value'),         # ——— NEW
    *[Input(f"inward-tilt-slider-{i}", "value") for i in range(1,5)]
)
def update_figure(selected, datetime_str, lat, lon, tz_off,
                  source_model, led_distance_mm, led_divergence_deg, *vals):
    tilts = vals[:4]
    use_point_source = (source_model == 'led')
    # sensible fallbacks if the inputs are blank
    led_distance_mm   = 9400.0 if led_distance_mm   in (None, '') else float(led_distance_mm)
    led_divergence_deg= 6.5    if led_divergence_deg in (None, '') else float(led_divergence_deg)

    fig = go.Figure()

    for idx, key in enumerate(['H1', 'H2', 'H3', 'H4']):
        if key not in selected: continue
        data = compute_heliostat_data(
            key, tilts[idx], datetime_str, lat, lon, tz_off,
            use_point_source=use_point_source,
            led_distance_mm=led_distance_mm,
            led_divergence_deg=led_divergence_deg
        )

        m = data["mesh"]
        if m["verts"].size:
            fig.add_trace(go.Mesh3d(
                x=m["verts"][:, 0], y=m["verts"][:, 1], z=m["verts"][:, 2],
                i=m["i"], j=m["j"], k=m["k"],
                opacity=0.5, color=m["color"], showscale=False,
                name=f'{key} Mirror'
            ))

        for start, end in data["incoming"]:
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode='lines', line=dict(color='orange', width=4), showlegend=False
            ))

        for start, end in data["reflected"]:
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode='lines', line=dict(color='green', width=4), showlegend=False
            ))

        for shaft, wing in data["normals"]:
            fig.add_trace(go.Scatter3d(
                x=[shaft[0], wing[0]], y=[shaft[1], wing[1]], z=[shaft[2], wing[2]],
                mode='lines', line=dict(color='green', width=6), showlegend=False
            ))

        hits = np.array(data["hits"])
        if hits.size:
            fig.add_trace(go.Scatter3d(
                x=hits[:, 0], y=hits[:, 1], z=hits[:, 2],
                mode='markers', marker=dict(color=m["color"], size=6),
                name=f'{key} Hits'
            ))

        sh = np.array(data["surface_hits_ring"])
        if sh.size:
            fig.add_trace(go.Scatter3d(
                x=sh[:, 0], y=sh[:, 1], z=sh[:, 2],
                mode='markers', marker=dict(size=2, color=m["color"]),
                name=f"{key} surface hits"
            ))

    # Receiver plane & centre
    xx, yy = np.meshgrid([-150,150],[-100,100])
    zz = np.full_like(xx, RECEIVER_POS[2])
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz, showscale=False, opacity=0.8,
        colorscale=[[0,'gray'],[1,'gray']], name='Receiver Plane'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[RECEIVER_POS[2]], mode='markers',
        marker=dict(symbol='cross',size=8), name='Receiver Center'
    ))

    fig.update_layout(
        scene=dict(aspectmode='data', xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)'),
        scene_camera=dict(eye=dict(x=0, y=0, z=2), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(itemsizing='constant')
    )
    return fig

if __name__ == "__main__":
    app.run(debug=True, port=8050)
