import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import math
import csv
from collections import defaultdict


model = YOLO("/content/best.pt")


in_zones_config = {
    "1in": np.array([[857, 896], [988, 752], [1067, 901], [942, 999]], dtype=np.int32),
    "2in": np.array([[861, 181], [765, 61], [866, 13], [953, 103]], dtype=np.int32),
    "3in": np.array([[210, 177], [83, 271], [26, 164], [138, 87]], dtype=np.int32),
    "4in": np.array([[155, 833], [400, 986], [173, 1052], [81, 918]], dtype=np.int32),
}

out_zones_config = {
    "1out": np.array([[905, 1041], [719, 1076], [628, 1028], [807, 914]], dtype=np.int32),
    "2out": np.array([[980, 157], [1078, 269], [1012, 387], [914, 219]], dtype=np.int32),
    "3out": np.array([[437, 52], [271, 131], [201, 57], [326, 2]], dtype=np.int32),
    "4out": np.array([[39, 619], [116, 822], [4, 901], [0, 678]], dtype=np.int32),
}
all_zones = {**in_zones_config, **out_zones_config}


zone_colors = {
    "1in": (0, 255, 0), "2in": (255, 255, 0), "3in": (255, 165, 0), "4in": (0, 255, 255),
    "1out": (0, 0, 255), "2out": (255, 0, 255), "3out": (128, 0, 128), "4out": (0, 128, 255)
}


active_tracks = {}
completed_trips = {}             
trip_speeds = defaultdict(list)   
completed_tracks = set()         


vehicle_records = {}

track_history = defaultdict(list)
speed_history = defaultdict(list)
METERS_PER_PIXEL = 0.04  


def draw_dashboard(frame, total_tracked, active_count, completed_count, avg_speed):
    if avg_speed > 17:
        status_text, status_color = "SMOOTH", (0, 255, 0)
    elif avg_speed > 8:
        status_text, status_color = "MODERATE", (0, 255, 255)
    else:
        status_text, status_color = "HEAVY", (0, 0, 255)

  
    x, y, w, h = 20, 20, 390, 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)

 
    cv2.putText(frame, "SYSTEM LIVE ANALYTICS", (x + 12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.line(frame, (x + 8, y + 36), (x + w - 8, y + 36), (100, 100, 100), 1)

    stats = [
        (f"Total Tracked : {total_tracked}", (220, 220, 220)),
        (f"Active On-Site: {active_count}", (0, 255, 255)),
        (f"Completed Trips: {completed_count}", (255, 200, 0)),
        (f"Avg Speed     : {int(avg_speed)} km/h", (255, 255, 255))
    ]

    
    start_y = y + 66
    for text, color in stats:
        cv2.putText(frame, text, (x + 12, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
        start_y += 28

    
    cv2.putText(frame, "Traffic Status:", (x + 12, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2)
    cv2.putText(frame, status_text, (x + 185, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)


video_path = "/content/IMG_20260830_002502_226.mp4"
output_path = "/content/output_OD_analytics.mp4"
csv_path = "/content/traffic_analytics_report.csv"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


results = model.track(
    source=video_path,
    conf=0.75,
    imgsz=640,
    tracker="bytetrack.yaml",
    stream=True
)

for r in results:
    frame = r.orig_img.copy()

    
    for z_name, pts in all_zones.items():
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], zone_colors[z_name])
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [pts], True, zone_colors[z_name], 2)

    detections = sv.Detections.from_ultralytics(r)

    if detections.tracker_id is not None and len(detections.tracker_id) > 0:
        for box, track_id, conf in zip(detections.xyxy, detections.tracker_id, detections.confidence):
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            current_zone = None
            for z_name, pts in all_zones.items():
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    current_zone = z_name
                    break

            if track_id in active_tracks:
                origin_zone = active_tracks[track_id]['origin']
                active_tracks[track_id]['path'].append((cx, cy))

                track_history[track_id].append((cx, cy))
                current_speed_kmh = 0
                if len(track_history[track_id]) > 5:
                    prev_x, prev_y = track_history[track_id][-5]
                    pixel_distance = math.hypot(cx - prev_x, cy - prev_y)
                    meters_distance = pixel_distance * METERS_PER_PIXEL

                    time_elapsed = 5 / fps
                    speed_mps = meters_distance / time_elapsed
                    calc_speed = speed_mps * 3.6

                    speed_history[track_id].append(calc_speed)
                    if len(speed_history[track_id]) > 5:
                        speed_history[track_id].pop(0)
                    current_speed_kmh = sum(speed_history[track_id]) / len(speed_history[track_id])

                
                if track_id in vehicle_records:
                    vehicle_records[track_id]['detection_count'] += 1
                    if current_speed_kmh > 0:
                        vehicle_records[track_id]['speeds'].append(current_speed_kmh)

                if current_zone and current_zone != origin_zone:
                   
                    trip_pair = (origin_zone, current_zone)
                    completed_trips[trip_pair] = completed_trips.get(trip_pair, 0) + 1

                    if current_speed_kmh > 0:
                        trip_speeds[trip_pair].append(current_speed_kmh)

                    
                    if track_id in vehicle_records:
                        vehicle_records[track_id]['destination'] = current_zone
                        vehicle_records[track_id]['status'] = 'Completed'

                    completed_tracks.add(track_id)
                    del active_tracks[track_id]
                    if track_id in speed_history:
                        del speed_history[track_id]
                else:
                    color = active_tracks[track_id]['color']
                    path = active_tracks[track_id]['path']

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if len(path) > 1:
                        path_pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [path_pts], False, color, 3)

                    label = f"#{track_id} | {int(current_speed_kmh)} km/h"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            else:
                
                if track_id in completed_tracks:
                    continue

                if current_zone:
                    active_tracks[track_id] = {
                        'origin': current_zone,
                        'color': zone_colors[current_zone],
                        'path': [(cx, cy)]
                    }
                    track_history[track_id].append((cx, cy))

                    
                    if track_id not in vehicle_records:
                        vehicle_records[track_id] = {
                            'origin': current_zone,
                            'destination': 'None',
                            'status': 'Active_In_Site',
                            'speeds': [],
                            'detection_count': 1
                        }

                    color = zone_colors[current_zone]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"#{track_id} | 0 km/h"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    
    for dest_name, dest_pts in all_zones.items():
        origin_counts = {orig: count for (orig, dest), count in completed_trips.items() if dest == dest_name}

        if origin_counts:
            M = cv2.moments(dest_pts)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                box_size = 24
                spacing = 4
                total_height = len(origin_counts) * (box_size + spacing)
                start_y = cy - total_height // 2

                for i, (orig, count) in enumerate(origin_counts.items()):
                    orig_color = zone_colors[orig]

                    x1_box = cx - box_size // 2
                    y1_box = start_y + i * (box_size + spacing)
                    x2_box = x1_box + box_size
                    y2_box = y1_box + box_size

                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), orig_color, -1)
                    cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (255, 255, 255), 1)

                    text = str(count)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    tx = x1_box + (box_size - tw) // 2
                    ty = y1_box + (box_size + th) // 2 - 2

                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    total_tracked = len(track_history)
    active_count = len(active_tracks)
    completed_count = sum(completed_trips.values())

    active_speeds = [
        sum(speed_history[tid]) / len(speed_history[tid])
        for tid in active_tracks if tid in speed_history and len(speed_history[tid]) > 0
    ]
    avg_speed = sum(active_speeds) / len(active_speeds) if active_speeds else 0

    draw_dashboard(frame, total_tracked, active_count, completed_count, avg_speed)

    out.write(frame)

cap.release()
out.release()


in_names = sorted(list(in_zones_config.keys()))
out_names = sorted(list(out_zones_config.keys()))

total_completed = sum(completed_trips.values())

with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    
    writer.writerow(['=== SECTION 1: INDIVIDUAL VEHICLE TRACKING LOG ==='])
    writer.writerow(['Vehicle_ID', 'Origin_Zone', 'Destination_Zone', 'Trip_Status', 'Avg_Speed_KMH', 'Detection_Count'])

    for vid in sorted(vehicle_records.keys()):
        rec = vehicle_records[vid]
        v_speeds = rec['speeds']
        v_avg_speed = (sum(v_speeds) / len(v_speeds)) if v_speeds else 0.0
        writer.writerow([
            vid,
            rec['origin'],
            rec['destination'],
            rec['status'],
            f"{v_avg_speed:.1f}",
            rec['detection_count']
        ])

    writer.writerow([])  

    
    writer.writerow(['=== SECTION 2: ORIGIN-DESTINATION (OD) MATRIX ==='])
    header_od = ['Origin / Destination'] + out_names + ['Total Origin']
    writer.writerow(header_od)

    col_totals = {out_z: 0 for out_z in out_names}

    for in_z in in_names:
        row = [in_z]
        row_sum = 0
        for out_z in out_names:
            cnt = completed_trips.get((in_z, out_z), 0)
            row.append(cnt)
            row_sum += cnt
            col_totals[out_z] += cnt
        row.append(row_sum)
        writer.writerow(row)

    total_row = ['Total Destination'] + [col_totals[out_z] for out_z in out_names] + [total_completed]
    writer.writerow(total_row)

    writer.writerow([])  

    
    writer.writerow(['=== SECTION 3: ROUTE DETAILED STATISTICS ==='])
    writer.writerow(['Rank', 'Origin', 'Destination', 'Vehicles Count', 'Percentage (%)', 'Avg Speed (KM/H)'])

    sorted_routes = sorted(completed_trips.items(), key=lambda x: x[1], reverse=True)

    for rank, ((orig, dest), count) in enumerate(sorted_routes, 1):
        pct = (count / total_completed * 100) if total_completed > 0 else 0
        speeds = trip_speeds.get((orig, dest), [])
        r_avg_speed = sum(speeds) / len(speeds) if speeds else 0
        writer.writerow([rank, orig, dest, count, f"{pct:.2f}%", f"{r_avg_speed:.1f}"])

print(f"\n[DONE] Processing complete!")
print(f"Video saved to: {output_path}")
print(f"CSV Report saved to: {csv_path}")
