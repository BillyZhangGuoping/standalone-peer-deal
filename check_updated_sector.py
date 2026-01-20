import csv

def check_updated_sector():
    csv_file = "Market_Inform/all_instruments_info_with_sector.csv"
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 统计总行数
    total_rows = len(rows)
    print(f"总行数: {total_rows}")
    
    # 统计各sector的分布
    sector_counts = {}
    for row in rows:
        sector = row['sector']
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    print("\n各sector分布:")
    for sector, count in sector_counts.items():
        print(f"{sector}: {count}行")
    
    # 显示部分示例，包括前5行和后5行
    print("\n部分示例:")
    print("前5行:")
    for row in rows[:5]:
        print(f"{row['sec_id']} : {row['sector']}")
    
    print("\n最后5行:")
    for row in rows[-5:]:
        print(f"{row['sec_id']} : {row['sector']}")
    
    # 显示每个sector的1个示例
    print("\n每个sector的示例:")
    seen_sectors = set()
    for row in rows:
        sector = row['sector']
        if sector not in seen_sectors:
            print(f"{row['sec_id']} : {sector}")
            seen_sectors.add(sector)
        if len(seen_sectors) == len(sector_counts):
            break

if __name__ == "__main__":
    check_updated_sector()
