import numpy as np
import folium
import matplotlib.pyplot as plt
from folium.raster_layers import ImageOverlay
import geopandas as gpd
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from branca.element import Template, MacroElement

# ==== 1. Dữ liệu ban đầu ====
# for idx, pre in enumerate(timelamp):
def plot_map(filtered_df):
    idx = 'pre'
    vn_mainland = gpd.read_file("map/gadm41_VNM_1.json")
    hoang_sa = gpd.read_file("map/gadm36_XSP_0.json")
    truong_sa = gpd.read_file("map/gadm36_XPI_0.json")

    # Gộp lại thành một GeoDataFrame
    vn_full = gpd.GeoDataFrame(pd.concat([vn_mainland, hoang_sa, truong_sa], ignore_index=True), crs='EPSG:4326')
    m = folium.Map(location=[14.0583, 108.2772], zoom_start=6, tiles='CartoDB Positron')

    X = filtered_df[['Kinh độ', 'Vĩ độ']]
    y = filtered_df['AQI_PM2.5']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # === 2. Tạo lưới để dự đoán ===
    lat_min, lat_max = X['Kinh độ'].min()-1, X['Kinh độ'].max()+1
    lon_min, lon_max = X['Vĩ độ'].min()-1, X['Vĩ độ'].max()+1
    grid_lat = np.linspace(lat_min, lat_max, 300)
    grid_lon = np.linspace(lon_min, lon_max, 300)
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    grid_points = np.c_[grid_lat_mesh.ravel(), grid_lon_mesh.ravel()]
    grid_intensity = model.predict(grid_points).reshape(grid_lat_mesh.shape)
    grid_intensity = gaussian_filter(grid_intensity, sigma=2)

    vn_union = vn_full.union_all()  # Nối tất cả vùng lại thành 1 polygon

    # Tạo mask: điểm nào ngoài biên VN thì đặt NaN
    mask = np.full(grid_intensity.shape, False)

    for i in range(grid_lat_mesh.shape[0]):
        for j in range(grid_lat_mesh.shape[1]):
            point = Point(grid_lon_mesh[i, j], grid_lat_mesh[i, j])
            if not vn_union.contains(point):
                mask[i, j] = True

    grid_intensity_masked = np.ma.array(grid_intensity, mask=mask)

    plt.figure(figsize=(6, 6))
    # plt.imshow(grid_intensity_masked, extent=(lon_min, lon_max, lat_min, lat_max),
    #         origin='lower', cmap='hot', alpha=0.6)
    # Danh sách màu chuẩn AQI theo thứ tự tăng dần
    aqi_colors = [
        '#00e400',  # Green (0–50)
        '#ffff00',  # Yellow (51–100)
        '#ff7e00',  # Orange (101–150)
        '#ff0000',  # Red (151–200)
        '#8f3f97',  # Purple (201–300)
        '#7e0023'   # Maroon (301–500)
    ]

    # Tạo colormap nội suy từ danh sách trên
    aqi_cmap = LinearSegmentedColormap.from_list("aqi_smooth", aqi_colors, N=100)

    plt.imshow(
        grid_intensity_masked,
        extent=(lon_min, lon_max, lat_min, lat_max),
        origin='lower',
        cmap=aqi_cmap,     # Hoặc 'plasma', 'turbo', 'jet', 'YlOrRd'
        alpha=0.75,
        vmin=0,
        vmax=400
    )
    plt.axis('off')
    plt.savefig(f"images/heatmap_overlay {idx}.png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # ==== 4. Tạo bản đồ và chèn ảnh ====

    # Overlay ảnh lên đúng vùng địa lý
    ImageOverlay(
        image=f"images/heatmap_overlay {idx}.png",
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        opacity=0.6,
        interactive=True,
        cross_origin=False
    ).add_to(m)

        # Đọc và thêm biên giới VN nếu có
    try:
        folium.GeoJson(
            vn_full,
            name='Biên giới',
            style_function=lambda x: {
                'fillColor': 'none',
                'color': 'black',
                'weight': 1,
                'dashArray': '5, 5'
            }
        ).add_to(m)
    except:
        print("Không tìm thấy tệp GeoJSON biên giới.")

    def get_folium_color(aqi):
        if aqi <= 50:
            return 'green'
        elif aqi <= 100:
            return 'beige'      # gần giống yellow
        elif aqi <= 150:
            return 'orange'
        elif aqi <= 200:
            return 'red'
        elif aqi <= 300:
            return 'purple'
        else:
            return 'darkred'


    for name, lat, lon, aqi_value in zip(filtered_df["Tên"], filtered_df["Kinh độ"], filtered_df["Vĩ độ"], filtered_df["AQI_PM2.5"]):
        # aqi_value = intensity_values[list(coordinates.keys()).index(name)]

        fill_color = (
            'green'  if aqi_value <= 50 else
            'yellow' if aqi_value <= 100 else
            'orange' if aqi_value <= 150 else
            'red'    if aqi_value <= 200 else
            'purple' if aqi_value <= 300 else
            'maroon'
        )

        folium.Marker(
            location=(lat, lon),
            popup=f"{name}: AQI PM2.5 = {aqi_value:.1f}",
            icon=folium.Icon(color=get_folium_color(aqi_value), icon="info-sign")  # icon bạn có thể đổi thành 'cloud', 'leaf', 'home'...
        ).add_to(m)
        
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            popup=f"{name}: pm2.5 {aqi_value}",
            color='black',
            weight=0.5,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.8
        ).add_to(m)

    folium.Marker(
        location=[16.5053, 111.9537],
        icon=folium.DivIcon(
            html='<div style="font-size: 12px; color: black; font-weight: bold;">Hoàng Sa</div>'
        )
    ).add_to(m)

    folium.Marker(
        location=[9.9342, 114.3302],
        icon=folium.DivIcon(
            html='<div style="font-size: 12px; color: black; font-weight: bold;">Trường Sa</div>'
        )
    ).add_to(m)

    folium.LayerControl().add_to(m)
    # m.save(f"heatmap_image_overlay{idx}.html")
    print("✅ Đã tạo bản đồ heatmap với overlay hình ảnh: heatmap_image_overlay.html")

    colorbar_template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        width: 200px;
        height: 20px;
        z-index:9999;
        background: linear-gradient(to right, green, yellow, orange, red, purple, maroon);
        border: 1px solid black;
        text-align: center;
        font-size: 12px;
        color: black;">
        0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;50&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;100&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;150&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;200&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;300&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;500+
    </div>
    {% endmacro %}
    """

    colorbar = MacroElement()
    colorbar._template = Template(colorbar_template)
    m.get_root().add_child(colorbar)
    m.save(f"heatmap_image_overlaycolorbar.html")
    m
    return m