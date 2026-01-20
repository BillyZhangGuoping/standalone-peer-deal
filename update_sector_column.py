import json
import csv

def update_sector_column():
    # 读取instrument_sector.json文件
    sector_file = "Market_Inform/instrument_sector.json"
    csv_file = "Market_Inform/all_instruments_info_with_sector.csv"
    
    # 解析JSON文件
    with open(sector_file, 'r', encoding='utf-8') as f:
        sector_data = json.load(f)
    
    # 将新格式转换为品种名称到sector的映射
    sector_map = {}
    for sector, symbols in sector_data.items():
        for symbol in symbols:
            sector_map[symbol] = sector
    
    # 读取CSV文件
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    # 为每一行更新sector列
    for row in rows:
        sec_id = row['sec_id']
        # 匹配sector，如果没有匹配则填入'other'
        sector = sector_map.get(sec_id, sector_map.get(sec_id.lower(), 'other'))
        row['sector'] = sector
    
    # 将结果写回CSV文件
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"成功更新了{csv_file}的sector列")
    print(f"共处理{len(rows)}行数据")

if __name__ == "__main__":
    update_sector_column()
