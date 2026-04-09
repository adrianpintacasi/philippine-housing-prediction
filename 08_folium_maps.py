"""Generate three interactive Folium maps of Philippine property listings.

Produces:
  - map_clusters.html  (color by K-Means cluster)
  - map_prices.html    (color by price tier)
  - map_combined.html  (color by cluster, radius scaled by price)
"""
import pandas as pd
import folium
from sklearn.cluster import KMeans

df = pd.read_csv('PH_Housing_Cleaned.csv')

coords = df[['Latitude', 'Longitude']].values
km = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = km.fit_predict(coords)

center_lat = df['Latitude'].mean()
center_lon = df['Longitude'].mean()

cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']
cluster_labels = [
    'Metro Manila & S. Luzon',
    'Mindanao (Davao)',
    'Visayas (Cebu)',
    'Northern Luzon',
    'Central Luzon',
]

def price_color(price):
    if price < 5_000_000:
        return 'green'
    if price <= 20_000_000:
        return 'orange'
    return 'red'

def make_popup(row, extra=''):
    return folium.Popup(
        f"<b>{str(row['Description'])[:60]}</b><br>"
        f"Price: P{row['Price']:,.0f}<br>"
        f"{extra}"
        f"{row['Location']}",
        max_width=300
    )

# ---- Map 1: clusters ----
m1 = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')
for _, row in df.iterrows():
    c = int(row['Cluster'])
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4, color=cluster_colors[c], fill=True,
        fill_color=cluster_colors[c], fill_opacity=0.6,
        popup=make_popup(row, f"Cluster: {c+1} ({cluster_labels[c]})<br>")
    ).add_to(m1)

legend1 = '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;' \
          'background:white;padding:10px;border-radius:5px;border:2px solid grey;"><b>Clusters</b><br>'
for i, (color, label) in enumerate(zip(cluster_colors, cluster_labels)):
    legend1 += f'<i style="background:{color};width:12px;height:12px;display:inline-block;border-radius:50%;"></i> {i+1}: {label}<br>'
legend1 += '</div>'
m1.get_root().html.add_child(folium.Element(legend1))
m1.save('map_clusters.html')
print("Saved: map_clusters.html")

# ---- Map 2: price tiers ----
m2 = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')
for _, row in df.iterrows():
    color = price_color(row['Price'])
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4, color=color, fill=True, fill_color=color, fill_opacity=0.6,
        popup=make_popup(row, f"P{row['Price_per_sqm']:,.0f}/sqm<br>")
    ).add_to(m2)

legend2 = '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;' \
          'background:white;padding:10px;border-radius:5px;border:2px solid grey;"><b>Price Tier</b><br>' \
          '<i style="background:green;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Affordable (below P5M)<br>' \
          '<i style="background:orange;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Mid-Range (P5M-P20M)<br>' \
          '<i style="background:red;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Luxury (above P20M)<br></div>'
m2.get_root().html.add_child(folium.Element(legend2))
m2.save('map_prices.html')
print("Saved: map_prices.html")

# ---- Map 3: combined (cluster color, radius by price) ----
m3 = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')
price_norm = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())
for idx, row in df.iterrows():
    c = int(row['Cluster'])
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3 + price_norm.loc[idx] * 12,
        color=cluster_colors[c], fill=True, fill_color=cluster_colors[c], fill_opacity=0.5,
        popup=make_popup(
            row,
            f"P{row['Price_per_sqm']:,.0f}/sqm<br>"
            f"Floor: {row['Floor Area']}sqm | Land: {row['Land Area']}sqm<br>"
            f"Bed: {int(row['Bedrooms'])} | Bath: {int(row['Bathrooms'])}<br>"
            f"Cluster: {c+1} ({cluster_labels[c]})<br>"
        )
    ).add_to(m3)

m3.save('map_combined.html')
print("Saved: map_combined.html")
