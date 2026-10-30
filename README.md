# Non-traversable 2D1A reach-avoid game interactive website

## Overview

This directory contains a static website for exploring the reach-avoid research
code. It provides two interactive views:

- barrier geometry for circular and convex polygonal target regions;
- a two-defender, one-attacker simulation for a non-traversable circular target.

The site is fully static. Python runs in the browser through Pyodide, while the
interface and Canvas rendering use plain HTML, CSS, and JavaScript.

## Project structure

```text
reach_avoid_webpage/
├── index.html                Page structure
├── styles.css                Layout and visual styles
├── app.js                    UI, Canvas rendering, and Pyodide bridge
├── scripts/
│   ├── api/                  Browser-facing Python API
│   ├── barrier/              Circle and polygon barrier geometry
│   ├── simulation/           Simulation state and integration
│   └── strategy/             OTP and equilibrium strategy solvers
├── validation.py             Numerical regression checks
└── README.md
```

## Deployment

### Local preview

Start a static HTTP server from the repository root:

```bash
python -m http.server 8000
```

Open:

```text
http://localhost:8000/non-traversable-reach-avoid/
```
