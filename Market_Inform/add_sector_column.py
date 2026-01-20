import json
import csv

# 读取instrument_sector.json文件
def read_sector_map():
    with open('instrument_sector.json', 'r', encoding='utf-8') as f:
        sector_data = json.load(f)
    return sector_data['Contract-to-Sector Map']

# 读取all_instruments_info.csv文件，添加sector列并保存
def add_sector_column():
    # 读取sector映射
    sector_map = read_sector_map()
    
    # 读取CSV文件
    input_file = 'all_instruments_info.csv'
    output_file = 'all_instruments_info_with_sector.csv'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ['sector']
        rows = list(reader)
    
    # 遍历每一行，添加sector列
    for row in rows:
        sec_id = row['sec_id']
        # 使用sec_id在sector_map中查找对应的sector，转换为小写以匹配
        sector = sector_map.get(sec_id, sector_map.get(sec_id.lower(), 'other'))
        row['sector'] = sector
    
    # 保存到新的CSV文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"处理完成！新文件已保存到 {output_file}")
    print(f"共处理 {len(rows)} 行数据")

if __name__ == '__main__':
    add_sector_column()
