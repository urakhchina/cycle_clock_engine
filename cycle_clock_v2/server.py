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

Run: python server.py
"""

import asyncio
import json
import numpy as np
import websockets
from engine.game import Game
from engine.cycle_clock import ISVParams


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


game = None


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


async def handle(websocket):
    print(f"Client connected")
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get('cmd')

            if cmd == 'get_board':
                board = get_board_data()
                await websocket.send(json.dumps({'type': 'board', 'data': board}, cls=NpEncoder))

            elif cmd == 'get_state':
                state = game.get_state()
                await websocket.send(json.dumps({'type': 'state', 'data': state}, cls=NpEncoder))

            elif cmd == 'step':
                n = data.get('n', 1)
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
                    'c0_segments': list(emp0 - overlap),       # C0 only
                    'c1_segments': list(emp1 - overlap),       # C1 only
                    'overlap_segments': list(overlap),          # shared
                    'c0_size': len(emp0),
                    'c1_size': len(emp1),
                    'overlap_size': len(overlap),
                }
                await websocket.send(json.dumps({'type': 'step', 'data': state}, cls=NpEncoder))

            elif cmd == 'step_with_options':
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
                game.update_isv(clock_id, params)
                await websocket.send(json.dumps({
                    'type': 'isv_updated',
                    'clock': clock_id,
                    'params': params
                }))

            elif cmd == 'reset':
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
    init_game()

    # Start HTTP server for static files in a thread
    import http.server, threading, os
    viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viz')
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=viz_dir, **kwargs)
    httpd = http.server.HTTPServer(('localhost', 8766), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Static server: http://localhost:8766")

    print(f"WebSocket server: ws://localhost:8765")
    print(f"\nOpen http://localhost:8766 in your browser\n")
    async with websockets.serve(handle, "localhost", 8765):
        await asyncio.Future()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped.")
