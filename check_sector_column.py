import csv

def check_sector_column():
    csv_file = "Market_Inform/all_instruments_info.csv"
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 统计总行数
    total_rows = len(rows)
    print(f"总行数: {total_rows}")
    
    # 统计other数量
    other_count = sum(1 for row in rows if row['sector'] == 'other')
    print(f"other数量: {other_count}")
    
    # 显示部分示例，包括最后5行
    print("\n部分示例:")
    print("前5行:")
    for row in rows[:5]:
        print(f"{row['sec_id']} : {row['sector']}")
    
    print("\n最后5行:")
    for row in rows[-5:]:
        print(f"{row['sec_id']} : {row['sector']}")
    
    # 显示some other的例子
    print("\nOther示例:")
    other_rows = [row for row in rows if row['sector'] == 'other']
    for row in other_rows[:5]:
        print(f"{row['sec_id']} : {row['sector']}")

if __name__ == "__main__":
    check_sector_column()
