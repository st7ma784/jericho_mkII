#!/usr/bin/env python3
"""
Real-time web visualization server for Jericho Mk II
Monitors simulation output and streams data to web clients via WebSockets
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from aiohttp import web
import aiohttp_cors

# Global state
simulation_state = {
    "running": False,
    "current_step": 0,
    "time": 0.0,
    "output_dir": None,
    "last_update": 0,
}

connected_clients = set()


class SimulationMonitor:
    """Monitors simulation output directory for new data"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.last_step = -1
        self.diagnostics = []

    def get_latest_step(self) -> int:
        """Find the most recent timestep available"""
        if not self.output_dir.exists():
            return -1

        field_files = list(self.output_dir.glob("fields_*.h5"))
        if not field_files:
            return -1

        steps = []
        for f in field_files:
            try:
                step = int(f.stem.split("_")[1])
                steps.append(step)
            except (IndexError, ValueError):
                continue

        return max(steps) if steps else -1

    def read_fields(self, step: int) -> Optional[Dict]:
        """Read electromagnetic field data for a given timestep"""
        field_file = self.output_dir / f"fields_{step:06d}.h5"

        if not field_file.exists():
            return None

        try:
            with h5py.File(field_file, "r") as f:
                # Read field arrays and downsample for web display
                Ex = np.array(f["Ex"])
                Ey = np.array(f["Ey"])
                Bz = np.array(f["Bz"])

                # Downsample if too large
                if Ex.shape[0] > 512 or Ex.shape[1] > 512:
                    stride_x = max(1, Ex.shape[0] // 512)
                    stride_y = max(1, Ex.shape[1] // 512)
                    Ex = Ex[::stride_x, ::stride_y]
                    Ey = Ey[::stride_x, ::stride_y]
                    Bz = Bz[::stride_x, ::stride_y]

                # Compute derived quantities
                E_mag = np.sqrt(Ex**2 + Ey**2)
                B_mag = np.abs(Bz)

                return {
                    "step": step,
                    "Ex": Ex.tolist(),
                    "Ey": Ey.tolist(),
                    "Bz": Bz.tolist(),
                    "E_magnitude": E_mag.tolist(),
                    "B_magnitude": B_mag.tolist(),
                    "shape": Ex.shape,
                }
        except Exception as e:
            print(f"Error reading fields: {e}")
            return None

    def read_particles(self, step: int, max_particles: int = 5000) -> Optional[Dict]:
        """Read particle data for visualization (downsampled)"""
        particle_file = self.output_dir / f"particles_{step:06d}.h5"

        if not particle_file.exists():
            return None

        try:
            with h5py.File(particle_file, "r") as f:
                x = np.array(f["x"])
                y = np.array(f["y"])
                vx = np.array(f["vx"])
                vy = np.array(f["vy"])
                vz = np.array(f["vz"])
                types = np.array(f["type"])

                n_particles = len(x)

                # Downsample if too many particles
                if n_particles > max_particles:
                    indices = np.random.choice(
                        n_particles, max_particles, replace=False
                    )
                    x = x[indices]
                    y = y[indices]
                    vx = vx[indices]
                    vy = vy[indices]
                    vz = vz[indices]
                    types = types[indices]

                # Compute velocity magnitude
                v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

                return {
                    "step": step,
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "vx": vx.tolist(),
                    "vy": vy.tolist(),
                    "vz": vz.tolist(),
                    "v_magnitude": v_mag.tolist(),
                    "type": types.tolist(),
                    "count": len(x),
                    "total_count": n_particles,
                }
        except Exception as e:
            print(f"Error reading particles: {e}")
            return None

    def read_diagnostics(self) -> List[Dict]:
        """Read diagnostics CSV file"""
        diag_file = self.output_dir / "diagnostics.csv"

        if not diag_file.exists():
            return []

        try:
            import csv

            diagnostics = []
            with open(diag_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert to float where possible
                    diag_row = {}
                    for key, value in row.items():
                        try:
                            diag_row[key] = float(value)
                        except (ValueError, TypeError):
                            diag_row[key] = value
                    diagnostics.append(diag_row)

            return diagnostics
        except Exception as e:
            print(f"Error reading diagnostics: {e}")
            return []


# WebSocket connection manager
async def websocket_handler(request):
    """Handle WebSocket connections for real-time updates"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    connected_clients.add(ws)
    print(f"Client connected. Total clients: {len(connected_clients)}")

    try:
        # Send initial state
        await ws.send_json(
            {"type": "state", "data": simulation_state}
        )

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    command = data.get("command")

                    if command == "set_output_dir":
                        simulation_state["output_dir"] = data.get("path")
                        await ws.send_json(
                            {"type": "ack", "message": "Output directory set"}
                        )

                    elif command == "request_data":
                        step = data.get("step", simulation_state["current_step"])
                        await send_simulation_data(ws, step)

                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})

            elif msg.type == web.WSMsgType.ERROR:
                print(f"WebSocket error: {ws.exception()}")

    finally:
        connected_clients.remove(ws)
        print(f"Client disconnected. Total clients: {len(connected_clients)}")

    return ws


async def send_simulation_data(ws, step: int):
    """Send simulation data for a specific step to client"""
    if not simulation_state["output_dir"]:
        await ws.send_json({"type": "error", "message": "No output directory set"})
        return

    monitor = SimulationMonitor(simulation_state["output_dir"])

    # Read fields
    fields = monitor.read_fields(step)
    if fields:
        await ws.send_json({"type": "fields", "data": fields})

    # Read particles
    particles = monitor.read_particles(step)
    if particles:
        await ws.send_json({"type": "particles", "data": particles})

    # Read diagnostics
    diagnostics = monitor.read_diagnostics()
    if diagnostics:
        await ws.send_json({"type": "diagnostics", "data": diagnostics})


async def monitor_simulation(app):
    """Background task to monitor simulation directory"""
    while True:
        if simulation_state["output_dir"]:
            monitor = SimulationMonitor(simulation_state["output_dir"])
            latest_step = monitor.get_latest_step()

            if latest_step > simulation_state["current_step"]:
                simulation_state["current_step"] = latest_step
                simulation_state["running"] = True
                simulation_state["last_update"] = time.time()

                # Broadcast update to all clients
                for ws in connected_clients:
                    try:
                        await send_simulation_data(ws, latest_step)
                    except Exception as e:
                        print(f"Error sending to client: {e}")

            elif time.time() - simulation_state["last_update"] > 5:
                simulation_state["running"] = False

        await asyncio.sleep(0.5)  # Check every 500ms


# HTTP handlers
async def index_handler(request):
    """Serve main HTML page"""
    html_file = Path(__file__).parent / "index.html"
    return web.FileResponse(html_file)


async def api_status(request):
    """API endpoint for simulation status"""
    return web.json_response(simulation_state)


async def api_set_output(request):
    """API endpoint to set output directory"""
    data = await request.json()
    output_dir = data.get("output_dir")

    if output_dir and Path(output_dir).exists():
        simulation_state["output_dir"] = output_dir
        return web.json_response({"status": "ok", "output_dir": output_dir})
    else:
        return web.json_response(
            {"status": "error", "message": "Invalid directory"}, status=400
        )


async def start_background_tasks(app):
    """Start background monitoring task"""
    app["monitor_task"] = asyncio.create_task(monitor_simulation(app))


async def cleanup_background_tasks(app):
    """Cleanup background tasks"""
    app["monitor_task"].cancel()
    await app["monitor_task"]


def create_app():
    """Create and configure the web application"""
    app = web.Application()

    # Configure CORS
    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True, expose_headers="*", allow_headers="*"
            )
        },
    )

    # Routes
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_get("/api/status", api_status)
    app.router.add_post("/api/set_output", api_set_output)

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.router.add_static("/static/", static_dir)

    # Configure CORS on all routes
    for route in list(app.router.routes()):
        if not isinstance(route.resource, web.StaticResource):
            cors.add(route)

    # Background tasks
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Jericho Mk II Real-time Visualization Server"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output",
        help="Simulation output directory to monitor",
    )
    parser.add_argument("--port", "-p", type=int, default=8888, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")

    args = parser.parse_args()

    simulation_state["output_dir"] = args.output_dir

    print(f"Starting Jericho Mk II Visualization Server")
    print(f"Monitoring: {args.output_dir}")
    print(f"Server: http://{args.host}:{args.port}")

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
