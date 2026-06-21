import os
import folium
from folium.plugins import HeatMap

# Fixed Mumbai Locations (area centres)
MUMBAI_LOCATIONS = {
    "Andheri": (19.1197, 72.8468),
    "Bandra": (19.0544, 72.8406),
    "Kurla": (19.0726, 72.8826),
    "Dadar": (19.0178, 72.8478),
    "Worli": (19.0176, 72.8174),
    "Powai": (19.1176, 72.9060),
    "Chembur": (19.0522, 72.9005),
    "Borivali": (19.2307, 72.8567),
    "Thane": (19.2183, 72.9781),
    "CST": (18.9398, 72.8355),
    "Ghatkopar": (19.0856, 72.9081),
    "Mulund": (19.1726, 72.9566),
    "Sion": (19.0434, 72.8610),
    "Malad": (19.1864, 72.8484),
    "Vashi": (19.0771, 72.9986),
}

_AREA_LIST = list(MUMBAI_LOCATIONS.values())


def _stable_offset(seed: int):
    """Deterministic small lat/lon jitter (~±1.2 km) for one violation.

    Uses a hash so a given row always lands on the same spot — the map does not
    jump on every rerun — while spreading many violations into a dense scatter
    instead of stacking them on a single area centre.
    """
    h = (seed * 2654435761) & 0xFFFFFFFF
    dlat = ((h & 0xFFFF) / 0xFFFF - 0.5) * 0.022
    dlon = (((h >> 16) & 0xFFFF) / 0xFFFF - 0.5) * 0.022
    return dlat, dlon


def _resolve_point(location: str, seed: int):
    """Pick a base area centre, jittered. Unknown/legacy locations are spread
    across the city so historical rows still populate the map."""
    if location in MUMBAI_LOCATIONS:
        base = MUMBAI_LOCATIONS[location]
    else:
        base = _AREA_LIST[seed % len(_AREA_LIST)]
    dlat, dlon = _stable_offset(seed)
    return base[0] + dlat, base[1] + dlon


def generate_heatmap(df, output_path="output/heatmaps/violation_heatmap.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Mumbai bounds
    south, west, north, east = 18.88, 72.75, 19.32, 73.05

    # Real street map (OpenStreetMap) for the "actual map" look.
    m = folium.Map(
        location=[19.0760, 72.8777],
        zoom_start=12,
        min_zoom=11,
        max_zoom=17,
        max_bounds=True,
        control_scale=True,
        tiles="OpenStreetMap",
    )
    m.fit_bounds([[south, west], [north, east]])

    heat_data = []

    for i, (_, row) in enumerate(df.iterrows()):
        location = str(row.get("Location", "")).strip()
        lat, lon = _resolve_point(location, i)
        heat_data.append([lat, lon])

        vehicle = row.get("Vehicle Number", "Unknown")
        violation = row.get("Violation Type", "Unknown")
        timestamp = row.get("Timestamp", "")
        popup_html = (
            f"<b>Location:</b> {location or 'Unknown'}<br>"
            f"<b>Vehicle:</b> {vehicle}<br>"
            f"<b>Violation:</b> {violation}<br>"
            f"<b>Time:</b> {timestamp}"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#d10000",
            fill=True,
            fill_color="#ff1a1a",
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(popup_html, max_width=240),
        ).add_to(m)

    # Soft density layer underneath the dots emphasises hotspots.
    if heat_data:
        HeatMap(heat_data, radius=16, blur=14, min_opacity=0.25).add_to(m)

    # Keep the viewport pinned to Mumbai.
    m.get_root().html.add_child(folium.Element(f"""
    <script>
    var bounds = L.latLngBounds(
        L.latLng({south}, {west}),
        L.latLng({north}, {east})
    );
    setTimeout(function(){{
        map.fitBounds(bounds);
        map.setMaxBounds(bounds);
    }}, 500);
    </script>
    """))

    m.save(output_path)
    return output_path
