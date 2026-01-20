import pandas as pd
import os


def save_daily_target_positions(daily_target_positions, date, output_dir, logger=None):
    """
    保存每日目标头寸到文件
    
    参数:
    - daily_target_positions: 每日目标头寸列表
    - date: 日期对象
    - output_dir: 输出目录
    - logger: 日志记录器（可选）
    """
    if daily_target_positions:
        positions_df = pd.DataFrame(daily_target_positions)
        # 只保留position_size不为0的品种
        positions_df = positions_df[positions_df['position_size'] != 0]
        
        # 保存到文件
        positions_file = os.path.join(output_dir, f'target_positions_{date.strftime("%Y%m%d")}.csv')
        positions_df.to_csv(positions_file, index=False)
        
        if logger:
            logger.info(f"目标头寸已保存到 {positions_file}")
            logger.info(f"生成了 {len(positions_df)} 个品种的目标头寸")
        
        return positions_file
    
    return None
