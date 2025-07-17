import time
import os
from datetime import datetime, timezone, timedelta
import requests

# 牧場設定字典
# 格式：{牧場名稱: {主機位址, 輸出目錄}}
FARM_SETTINGS = {
        "115": {
        "host": "http://192.168.115.190:5000/",
        "output_dir": "115"
    },
    "50": {
        "host": "http://192.168.50.190:5000/",
        "output_dir": "50"
    },

    "109": {
        "host": "http://192.168.109.190:5000/",
        "output_dir": "109"
    },
    "116": {
        "host": "http://192.168.116.190:5000/",
        "output_dir": "116"
    },
    "114": {
        "host": "http://192.168.114.190:5000/",
        "output_dir": "114"
    },
    "110": {
        "host": "http://192.168.110.190:5000/",
        "output_dir": "110"
    },
    "111": {
        "host": "http://192.168.111.190:5000/",
        "output_dir": "111"
    },
    "107": {
        "host": "http://192.168.107.100:5000/",
        "output_dir": "107"
    },
    "117": {
        "host": "http://192.168.117.190:5000/",
        "output_dir": "117"
    },
    "105": {
        "host": "http://192.168.105.190:5000/",
        "output_dir": "105"
    },
    "106": {
        "host": "http://192.168.106.190:5000/",
        "output_dir": "106"
    },
    "108": {
        "host": "http://192.168.108.190:5000/",
        "output_dir": "108"
    },
    "112": {
        "host": "http://192.168.112.190:5000/",
        "output_dir": "112"
    }
}

# 確保所有牧場的輸出目錄都存在
for farm_setting in FARM_SETTINGS.values():
    os.makedirs(farm_setting["output_dir"], exist_ok=True)

# 遍歷每個牧場
for farm_name, farm_setting in FARM_SETTINGS.items():
    try:
        # 獲取攝影機清單並去除 _main/_sub 後綴
        response = requests.get(f"{farm_setting['host']}/api/config")
        cameras_raw = response.json()["cameras"]
        cameras = []
        for camera in cameras_raw.keys():
            # 去除 _main 或 _sub 後綴
            base_name = camera.replace("_main", "").replace("_sub", "")
            if base_name not in cameras:
                cameras.append(base_name)

        # 為每個攝影機抓取一張照片
        for camera in cameras:
            try:
                # 獲取截圖
                timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
                snapshot_url = f"{farm_setting['host']}/api/{camera}/latest.jpg"
                response = requests.get(snapshot_url)
                
                if response.status_code == 200:
                    # 建立輸出路徑
                    output_path = os.path.join(farm_setting["output_dir"], f"{camera}_{timestamp}.jpg")
                    # 儲存照片
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    print(f"已儲存 {farm_name} 的 {camera} 照片於 {timestamp}")
                else:
                    print(f"無法獲取 {farm_name} 的 {camera} 照片: HTTP {response.status_code}")
            except Exception as e:
                print(f"處理 {farm_name} 的 {camera} 時發生錯誤: {e}")
    except Exception as e:
        print(f"處理 {farm_name} 時發生錯誤: {e}")

print("所有照片抓取完成") 