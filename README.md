# 🚦 PathVision — AI-Powered Vehicle Tracking & Origin-Destination Analysis

An intelligent computer vision system that turns ordinary CCTV/traffic cameras into smart sensors. It detects vehicles, tracks their trajectories through intersections or roundabouts, estimates real-world speed, and generates Origin-Destination (OD) traffic reports to support transportation engineering decisions.

---

## 🎥 Demo

https://youtube.com/shorts/KgDTsUhquNM?si=BD0WAETnwr7GeXfZ
---

## ✨ Features

- Robust Vehicle Tracking — unique ID per vehicle, trajectory drawing, and smart memory (completed_tracks) to avoid duplicate counting or premature data loss.
- Real-World Speed Estimation — converts pixel displacement to meters (METERS_PER_PIXEL) and combines it with FPS to compute speed in km/h.
- Zone-Based Analytics — 4 entry zones + 4 exit zones as transparent colored polygons for precise route identification.
- Origin-Destination (OD) Matrix — tracks each vehicle's entry and exit zone and aggregates traffic flow between them.
- Live Dashboard Overlay — a compact, non-intrusive HUD on the video feed showing:
  - Total vehicles tracked
  - Vehicles currently active on-site
  - Completed trips
  - Average speed
- Traffic Status Indicator — dynamic classification: SMOOTH / MODERATE / HEAVY, color-coded.
- Smart Visual Counters — live counts above each exit zone, color-matched to their originating entry zone.
- CSV Report Export — three structured sections:
  1. Per-vehicle log (ID, origin, destination, status, avg speed, detection count)
  2. OD matrix (entry → exit flow volume)
  3. Top routes ranked by usage % and average speed

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | Real-time vehicle detection (custom best.pt weights) |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Multi-object tracking, ID persistence through occlusion |
| [Supervision](https://github.com/roboflow/supervision) | Detection/tracking output handling |
| OpenCV | Video I/O, polygon overlays, drawing, HUD rendering |
| NumPy | Zone coordinate arrays, geometric operations |
| Python math | Euclidean distance → speed calculation |
| Python csv, collections | Data aggregation (defaultdict, set) and report export |

---

## 📊 Output

Running the script generates a CSV report (report.csv) containing:
- Individual vehicle records
- OD flow matrix
- Top routes summary

---

## ⚠️ Limitations

- Speed accuracy depends on manual camera calibration (METERS_PER_PIXEL); no automatic perspective correction.
- Zones are currently hardcoded per camera view — not portable across setups without reconfiguration.
- No benchmark numbers (FPS, accuracy vs. ground truth) published yet.

---

## 🗺️ Roadmap

- [ ] Automatic zone calibration
- [ ] Multi-camera support
- [ ] Web dashboard for live monitoring
- [ ] Accuracy benchmarking against ground-truth radar/loop data

---

## 📄 License

MIT License

---

## 🤝 Contributing

Pull requests welcome. Open an issue first to discuss major changes.



## 📥 Model Weights

Download best.pt from [GitHub Releases]([your-release-link-here](https://github.com/anmarw38-ops/Pathvision/releases/download/v1.0/best.pt)) and place it in the project root before running main.py.
