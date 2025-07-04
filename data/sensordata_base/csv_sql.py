import sqlite3
import pandas as pd
import os
from typing import List

DB_PATH = "sensor_data.db" 


def init_db():
    """初始化数据库，创建数据表"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        oxygen_saturation REAL,
        water_level REAL,
        ph REAL,
        ph_temp REAL,
        turbidity REAL,
        turbidity_temp REAL
    )
    """)

    conn.commit()
    conn.close()
    print("[INFO] 数据库初始化完成")


def clear_database():
    """清空数据库中的所有数据"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("DELETE FROM sensor_data")
    conn.commit()
    cur.execute("DELETE FROM sqlite_sequence WHERE name='sensor_data'")
    conn.commit()   
    conn.close()
    print("[INFO] 数据库已清空")


def import_csv_to_db(csv_path: str):
    """批量导入CSV数据"""
    if not os.path.exists(csv_path):
        print(f"[ERROR] 文件不存在: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["时间"])
    df = df.rename(columns={
        "时间": "timestamp",
        "溶解氧饱和度": "oxygen_saturation",
        "液位(mm)": "water_level",
        "PH": "ph",
        "PH温度(°C)": "ph_temp",
        "浊度(NTU)": "turbidity",
        "浊度温度(°C)": "turbidity_temp"
    })
    
    sensor_columns = ["timestamp", "oxygen_saturation", "water_level", "ph", "ph_temp", "turbidity", "turbidity_temp"]
    df = df.dropna(subset=sensor_columns, how='all')

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sensor_data", conn, if_exists="append", index=False)
    conn.close()

    print(f"[INFO] 已成功导入 {len(df)} 条数据")


def query_sensor_data_by_date(dates: List[str]) -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    placeholders = ",".join(["?"] * len(dates))
    sql = f"""
    SELECT timestamp, oxygen_saturation, water_level, ph, ph_temp, turbidity, turbidity_temp
    FROM sensor_data
    WHERE date(timestamp) IN ({placeholders})
    ORDER BY timestamp
    """

    cur.execute(sql, dates)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "无相关数据"

    result_lines = []
    for r in rows:
        # Handle None values properly with safe formatting
        try:
            oxygen = f"{r[1]:.2f}" if r[1] is not None else "N/A"
        except (TypeError, ValueError):
            oxygen = "N/A"
            
        water_level = f"{r[2]:.1f}" if r[2] is not None else "N/A"
        ph = f"{r[3]:.2f}" if r[3] is not None else "N/A"
        ph_temp = f"{r[4]:.1f}" if r[4] is not None else "N/A"
        turbidity = f"{r[5]:.1f}" if r[5] is not None else "N/A"
        turbidity_temp = f"{r[6]:.1f}" if r[6] is not None else "N/A"
        
        line = f"{r[0]} | 溶解氧: {oxygen} | 液位: {water_level} mm | PH: {ph} | PH温度: {ph_temp}°C | 浊度: {turbidity} NTU | 浊度温度: {turbidity_temp}°C"
        result_lines.append(line)
    
    result = "\n".join(result_lines)
    return result


if __name__ == "__main__":

    print(query_sensor_data_by_date(["2025-06-13"]))
