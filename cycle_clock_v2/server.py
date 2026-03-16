"""
WebSocket server for live cycle clock simulation.
Also serves static files for the viz.

The front-end sends commands, the server runs the Python engine
and streams state updates back in real time.

Commands from front-end:
  {"cmd": "step"}                         → advance one step
  {"cmd": "step", "n": 10}               → advance 10 steps
  {"cmd": "reset"}                        → reset simulation
  {"cmd": "set_isv", "clock": 0, "params": {"savings_exponent": 15}}
  {"cmd": "get_state"}                    → request current state
  {"cmd": "get_board"}                    → request full board data

Helix mode commands:
  {"cmd": "init_helix", "preset": "teeter_totter"}
  {"cmd": "init_helix", "config": {"emperor1": {...}, "emperor2": {...}, "exponent": 28}}
  {"cmd": "step"}                         → works for both modes
  {"cmd": "get_state"}                    → works for both modes

Run: python server.py
"""

import asyncio
import json
import numpy as np
import websockets
from engine.game import Game
from engine.cycle_clock import ISVParams
from engine.helix_game import HelixGame, PRESETS


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


game = None
helix_game = None
mode = 'segment'  # 'segment' or 'helix'


def init_game():
    global game
    game = Game(verbose=True)
    c0 = game.add_clock(
        fig_vertex=game.fig.origin_idx,
        coxeter_seed=0,
        isv=ISVParams(savings_exponent=28.0))
    # Find farthest vertex with decent empire
    dists = np.linalg.norm(
        game.fig.pos_3d - game.fig.pos_3d[game.fig.origin_idx], axis=1)
    candidates = [(i, dists[i]) for i in range(game.fig.n_vertices)
                  if game.empire.vertex_empire_sizes[i] > 120
                  and game.segs.degrees[i] >= 6]
    candidates.sort(key=lambda x: -x[1])
    far_v = candidates[0][0]
    c1 = game.add_clock(
        fig_vertex=far_v,
        coxeter_seed=100,
        isv=ISVParams(savings_exponent=28.0))
    print(f"Clocks: C0@v{c0.vertex} (center), C1@v{c1.vertex} (far)")


def get_board_data():
    """Full board data for initial front-end load."""
    return {
        'n_vertices': game.fig.n_vertices,
        'n_segments': game.segs.n_segments,
        'positions_3d': game.fig.pos_3d.tolist(),
        'perp_radii': game.fig.perp_radius.tolist(),
        'segments': game.segs.segments,
        'degrees': game.segs.degrees.tolist(),
        'vertex_empire_sizes': game.empire.vertex_empire_sizes.tolist(),
        'segment_empire_sizes': game.empire.segment_empire_sizes.tolist(),
        'origin_idx': game.fig.origin_idx,
    }


def init_helix_game(preset_name=None, config=None, empire_radius=8):
    """Initialize helix mode simulation."""
    global helix_game, mode
    mode = 'helix'

    helix_game = HelixGame(empire_radius=empire_radius, verbose=True)

    if preset_name:
        helix_game.init_from_preset(preset_name)
        print(f"Helix mode: preset '{preset_name}'")
    elif config:
        for key in ['emperor1', 'emperor2']:
            if key in config:
                cfg = config[key]
                pos = cfg.get('position', [0, 0, 0])
                if isinstance(pos, str) and pos.startswith('empire['):
                    idx = int(pos.split('[')[1].rstrip(']'))
                    pos = helix_game.empire[min(idx, len(helix_game.empire) - 1)]
                helix_game.add_emperor(
                    position=pos,
                    axis_group=cfg.get('axis_group', 3),
                    chirality=cfg.get('chirality', 'L'),
                    exponent=config.get('exponent', 28),
                )
        print(f"Helix mode: custom config")


async def handle(websocket):
    global mode, helix_game
    print(f"Client connected")

    # Wait for game to be initialized if server just started
    while game is None:
        await asyncio.sleep(0.5)
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get('cmd')

            if cmd == 'init_helix':
                preset = data.get('preset')
                config = data.get('config')
                empire_radius = data.get('empire_radius', 8)
                init_helix_game(preset_name=preset, config=config,
                              empire_radius=empire_radius)
                state = helix_game.get_state()
                await websocket.send(json.dumps({
                    'type': 'helix_init',
                    'data': state,
                    'presets': list(PRESETS.keys()),
                }, cls=NpEncoder))

            elif cmd == 'get_board':
                if mode == 'helix' and helix_game:
                    # Helix mode: send empire points + emperor info
                    await websocket.send(json.dumps({
                        'type': 'board',
                        'data': {
                            'mode': 'helix',
                            'empire_size': len(helix_game.empire),
                            'empire_positions': helix_game.empire[::2].tolist(),  # every 2nd point (~9K)
                            'emperors': [e.snapshot() for e in helix_game.emperors],
                        }
                    }, cls=NpEncoder))
                else:
                    board = get_board_data()
                    await websocket.send(json.dumps({'type': 'board', 'data': board}, cls=NpEncoder))

            elif cmd == 'get_state':
                if mode == 'helix' and helix_game:
                    state = helix_game.get_state()
                else:
                    state = game.get_state()
                await websocket.send(json.dumps({'type': 'state', 'data': state}, cls=NpEncoder))

            elif cmd == 'step':
                n = data.get('n', 1)

                if mode == 'helix' and helix_game:
                    for _ in range(n):
                        step_data = helix_game.step()
                    state = helix_game.get_state()
                    state['step_data'] = step_data
                    await websocket.send(json.dumps({
                        'type': 'step', 'data': state
                    }, cls=NpEncoder))
                else:
                    for _ in range(n):
                        step_data = game.step()
                    # Send step with empire segment IDs for visualization
                    for cd in step_data['clocks']:
                        cd.pop('all_options', None)
                    state = game.get_state()
                    state['step_data'] = step_data
                    # Add empire segments for both clocks
                    c0v = game.clocks[0].vertex
                    c1v = game.clocks[1].vertex
                    emp0 = game.empire.segment_empire[c0v]
                    emp1 = game.empire.segment_empire[c1v]
                    overlap = emp0 & emp1
                    state['empires'] = {
                        'c0_segments': list(emp0 - overlap),
                        'c1_segments': list(emp1 - overlap),
                        'overlap_segments': list(overlap),
                        'c0_size': len(emp0),
                        'c1_size': len(emp1),
                        'overlap_size': len(overlap),
                    }
                    await websocket.send(json.dumps({'type': 'step', 'data': state}, cls=NpEncoder))

            elif cmd == 'step_with_options':
                if mode == 'helix' and helix_game:
                    # In helix mode, step_with_options behaves like step
                    step_data = helix_game.step()
                    state = helix_game.get_state()
                    state['step_data'] = step_data
                    await websocket.send(json.dumps({
                        'type': 'step', 'data': state
                    }, cls=NpEncoder))
                else:
                    step_data = game.step()
                    state = game.get_state()
                    state['step_data'] = step_data
                    c0v = game.clocks[0].vertex
                    c1v = game.clocks[1].vertex
                    emp0 = game.empire.segment_empire[c0v]
                    emp1 = game.empire.segment_empire[c1v]
                    overlap = emp0 & emp1
                    state['empires'] = {
                        'c0_segments': list(emp0 - overlap),
                        'c1_segments': list(emp1 - overlap),
                        'overlap_segments': list(overlap),
                        'c0_size': len(emp0),
                        'c1_size': len(emp1),
                        'overlap_size': len(overlap),
                    }
                    await websocket.send(json.dumps({'type': 'step_full', 'data': state}, cls=NpEncoder))

            elif cmd == 'set_isv':
                clock_id = data.get('clock', 0)
                params = data.get('params', {})
                if mode == 'helix' and helix_game:
                    # Update emperor exponent in helix mode
                    if clock_id < len(helix_game.emperors):
                        emp = helix_game.emperors[clock_id]
                        if 'savings_exponent' in params:
                            emp.exponent = params['savings_exponent']
                        if 'chirality' in params:
                            emp.chirality = params['chirality']
                else:
                    game.update_isv(clock_id, params)
                await websocket.send(json.dumps({
                    'type': 'isv_updated',
                    'clock': clock_id,
                    'params': params
                }))

            elif cmd == 'reset':
                if mode == 'helix':
                    mode = 'segment'
                    helix_game = None
                init_game()
                board = get_board_data()
                state = game.get_state()
                await websocket.send(json.dumps({
                    'type': 'reset',
                    'board': board,
                    'state': state
                }, cls=NpEncoder))

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    import os
    from http import HTTPStatus

    # Railway sets PORT env var; locally use defaults
    ws_port = int(os.environ.get('PORT', 8765))
    ws_host = '0.0.0.0'
    http_port = int(os.environ.get('HTTP_PORT', 8766))
    is_railway = 'PORT' in os.environ or 'RAILWAY_ENVIRONMENT' in os.environ

    # Health check handler — responds to plain HTTP so Railway's probe passes
    async def process_request(path, headers):
        if path == "/" or path == "/health":
            return HTTPStatus.OK, [], b"OK\n"

    # Start WebSocket server FIRST so health check passes while game builds
    print(f"WebSocket server starting on {ws_host}:{ws_port}...")
    server = await websockets.serve(
        handle, ws_host, ws_port,
        process_request=process_request)
    print(f"WebSocket server listening (health check ready)")

    # Now build the game (takes ~15s)
    init_game()

    # Start HTTP server for static files (local dev only)
    if not is_railway:
        import http.server, threading
        viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viz')
        class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=viz_dir, **kwargs)
            def end_headers(self):
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                super().end_headers()
        httpd = http.server.HTTPServer(('localhost', http_port), NoCacheHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"Static server: http://localhost:{http_port}")
        print(f"\nOpen http://localhost:{http_port} in your browser\n")

    await asyncio.Future()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
